"""
Speaker diarization service using NVIDIA NeMo diarization model.

Model: nvidia/diar_sortformer_4spk-v1

Outputs normalized segments: [{start: float, end: float, speaker: str}]
"""

from __future__ import annotations

import asyncio
import tempfile
import json
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from loguru import logger
from pathlib import Path


DIAR_MODEL_NAME = "nvidia/diar_sortformer_4spk-v1"


@dataclass
class DiarSegment:
    start: float
    end: float
    speaker: str  # e.g., "SPK_1"


class DiarizationService:
    def __init__(self, model_name: str = DIAR_MODEL_NAME, max_speakers: int = 4):
        self.model_name = model_name
        self.max_speakers = max_speakers
        self._model = None
        self._speaker_model = None  # For speaker embeddings
        self._lock = asyncio.Lock()
        self._mode = None  # 'sortformer' | 'msdd' | 'clustering'
        
        # Diarization-optimized chunking: 10 minutes to prevent CUDA OOM
        self.chunk_duration = 10 * 60  # 10 minutes in seconds (safe for Sortformer)
        self.overlap_duration = 30.0    # 30 seconds overlap for safety
        self.similarity_threshold = 0.35  # Cosine similarity threshold for speaker matching (lowered to handle noisy/short segments)
        
        # H100 concurrency limits - optimized for large-scale processing
        # Each 5hr job uses ~30GB VRAM with 24-minute chunking
        # H100 79GB can handle 2-3 jobs simultaneously safely
        self.max_concurrent_jobs = 3  # Conservative for stability
        self._job_semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        
        # Enhanced diarization parameters
        self.min_duration_on = 0.3  # Shorter minimum segments for better resolution
        self.max_duration_off = 1.5  # Tighter gap detection
        self.confidence_threshold = 0.8  # Minimum confidence for reliable segments

    async def _ensure_model(self):
        """Load diarization model with fallback for different model types."""
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
        
        try:
            logger.info(f"Loading diarization model: {self.model_name}")
            
            # Try EncDecSpeakerLabelModel first (works for sortformer models like diar_sortformer_4spk-v1)
            try:
                import nemo.collections.asr as nemo_asr
                logger.info(f"Loading NeMo EncDecSpeakerLabelModel from HuggingFace: {self.model_name}")
                
                # Configure to use HuggingFace as source
                import os
                os.environ['NGC_NEMO_CACHE'] = '/root/.cache/huggingface'
                
                # Handle API changes - try different parameter combinations
                try:
                    # Try loading from HuggingFace directly without override_config first
                    self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                        self.model_name
                    )
                except TypeError as e:
                    if "override_config" in str(e):
                        # Try without override_config for older API
                        logger.warning(f"Removing 'override_config' parameter for compatibility: {e}")
                        self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                            self.model_name
                        )
                    elif "strict" in str(e):
                        # Remove strict parameter for newer NeMo versions
                        logger.warning(f"Removing 'strict' parameter for compatibility: {e}")
                        self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                            self.model_name
                        )
                    else:
                        raise
                
                self._mode = 'sortformer'
                logger.info("EncDecSpeakerLabelModel loaded successfully from HuggingFace")
                return
            except Exception as e:
                logger.warning(f"EncDecSpeakerLabelModel not available: {e}")
            
            # Try NeuralDiarizer (for MSDD models)
            try:
                from nemo.collections.asr.models import NeuralDiarizer  # type: ignore
                logger.info(f"Loading NeMo NeuralDiarizer: {self.model_name}")
                self._model = NeuralDiarizer.from_pretrained(self.model_name)
                self._mode = 'msdd'
                logger.info("NeuralDiarizer loaded successfully")
                return
            except Exception as e:
                logger.warning(f"NeuralDiarizer not available: {e}")
            
            # Attempt modern clustering diarizer
            try:
                from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer  # type: ignore
                logger.info(f"Loading NeMo ClusteringDiarizer: {self.model_name}")
                
                # Handle API changes for ClusteringDiarizer
                try:
                    self._model = ClusteringDiarizer.from_pretrained(self.model_name)
                except TypeError as e:
                    if "strict" in str(e):
                        logger.warning(f"Removing 'strict' parameter for ClusteringDiarizer: {e}")
                        # Try without strict parameter - this might need a different approach
                        self._model = ClusteringDiarizer.restore_from(self.model_name)
                    else:
                        raise
                
                self._mode = 'clustering'
                logger.info("ClusteringDiarizer loaded successfully")
                return
            except Exception as e:
                logger.warning(f"ClusteringDiarizer not available: {e}")

        except Exception as e:
            logger.error(f"Unexpected error loading diarization model: {e}")

        # Could not load any compatible model - use fallback
        logger.warning(f"Could not load diarization model {self.model_name}, using fallback single-speaker mode")
        self._model = None
        self._mode = 'fallback'
        logger.info("Fallback diarization mode enabled (single speaker)")

    async def run(self, audio_path: str, max_speakers: Optional[int] = None) -> List[DiarSegment]:
        """
        Run diarization on audio file with memory-optimized chunking.
        
        Strategy:
        - Single chunk (≤10 minutes): Direct diarization
        - Multi-chunk (>10 minutes): Chunked processing with speaker re-identification
        - Max support: 5 hours total audio duration
        - Concurrency limit: Max 3 simultaneous jobs to prevent H100 VRAM exhaustion
        """
        # H100 concurrency control - prevent VRAM exhaustion
        async with self._job_semaphore:
            logger.info(f"Starting diarization (slot available: {self.max_concurrent_jobs - self._job_semaphore._value}/{self.max_concurrent_jobs})")
            
            await self._ensure_model()
            ms = max_speakers or self.max_speakers

            try:
                # Get audio duration
                import soundfile as sf
                audio, sr = sf.read(audio_path)
                duration = len(audio) / sr
                
                # Single chunk: Direct diarization for ≤24 minutes  
                if duration <= self.chunk_duration:
                    logger.info(f"Single-chunk diarization for {duration/60:.1f}m audio (≤{self.chunk_duration/60:.0f}m)")
                    loop = asyncio.get_event_loop()
                    segments = await loop.run_in_executor(None, self._run_sync, audio_path, ms)
                    logger.info(f"Diarization complete: {len(segments)} segments, {len(set(s.speaker for s in segments))} speakers")
                    return segments
                
                # Multi-chunk: Use chunking with TitaNet re-identification (fallback for >5h audio)
                logger.info(f"Multi-chunk diarization for {duration/3600:.1f}h audio (>{self.chunk_duration/3600:.0f}h - very long audio)")
                
                # Generate chunks
                chunks = self._generate_chunks(duration)
                logger.info(f"Generated {len(chunks)} chunks ({self.chunk_duration/3600:.0f}h each, {self.overlap_duration/60:.0f}min overlap)")
                
                # Process each chunk
                chunk_results = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk['start']:.1f}s - {chunk['end']:.1f}s")
                    segments = await self._diarize_chunk(audio_path, chunk, ms)
                    chunk_results.append({
                        'chunk_idx': i,
                        'chunk': chunk,
                        'segments': segments
                    })
                    logger.info(f"Chunk {i+1} complete: {len(segments)} segments, {len(set(s.speaker for s in segments))} speakers")
                
                # Extract speaker embeddings from overlap regions for cross-chunk matching
                logger.info("Extracting speaker embeddings from overlaps...")
                embeddings = await self._extract_overlap_embeddings(audio_path, chunk_results)
                logger.info(f"Extracted {len(embeddings)} speaker embeddings")
                
                # Re-identify speakers across chunks using TitaNet embeddings
                logger.info("Re-identifying speakers across chunks with TitaNet...")
                merged_segments = await self._merge_and_reidentify(chunk_results, embeddings)
                
                num_speakers = len(set(seg.speaker for seg in merged_segments))
                logger.info(f"Multi-chunk diarization complete: {len(merged_segments)} segments, {num_speakers} speakers")
                
                return merged_segments
                
            except Exception as e:
                logger.error(f"Diarization failed for {audio_path}: {e}")
                raise

    def _run_sync(self, audio_path: str, max_speakers: int) -> List[DiarSegment]:
        """Blocking diarization call; returns normalized segments.

        This function wraps actual model inference; adjust to match NeMo API.
        """
        try:
            diar_output = None
            # Use specific code path based on the model type we loaded
            if self._mode == 'sortformer':
                # EncDecSpeakerLabelModel / SortformerEncLabelModel
                # This is the correct path for nvidia/diar_sortformer_4spk-v1
                logger.info(f"Running Sortformer diarization on {audio_path}")
                # The diarize method takes audio path(s) directly
                # Returns List[List[str]] where each inner list contains RTTM lines
                diar_output = self._model.diarize(audio=audio_path, batch_size=1, verbose=False)
                logger.info(f"Sortformer diarization completed, output type: {type(diar_output)}")
                
            elif self._mode == 'msdd':
                # MSDD NeuralDiarizer - try different parameter combinations
                logger.info(f"Running MSDD diarization on {audio_path} with max_speakers={max_speakers}")
                try:
                    # Try with num_speakers first
                    diar_output = self._model.diarize(paths2audio_files=[audio_path], num_speakers=max_speakers)
                except (TypeError, AttributeError) as e:
                    logger.debug(f"First attempt failed: {e}, trying alternative parameters")
                    try:
                        # Try with max_num_speakers
                        diar_output = self._model.diarize(paths2audio_files=[audio_path], max_num_speakers=max_speakers)
                    except (TypeError, AttributeError):
                        # Try without speaker count (let model auto-detect)
                        diar_output = self._model.diarize(paths2audio_files=[audio_path])
                        
            elif self._mode == 'clustering':
                # ClusteringDiarizer
                logger.info(f"Running clustering diarization on {audio_path} with max_speakers={max_speakers}")
                try:
                    diar_output = self._model.diarize(audio_file=audio_path, max_num_speakers=max_speakers)
                except TypeError:
                    diar_output = self._model.diarize(audio_file=audio_path, num_speakers=max_speakers)
            elif self._mode == 'fallback':
                # Fallback single-speaker mode
                logger.info(f"Using fallback single-speaker diarization for {audio_path}")
                import soundfile as sf
                audio, sr = sf.read(audio_path)
                duration = len(audio) / sr
                # Return single segment for entire audio
                return [DiarSegment(start=0.0, end=duration, speaker="SPK_1")]
            elif self._mode == 'rule_based':
                # Rule-based diarization - simple energy-based speaker detection
                logger.info(f"Using rule-based diarization for {audio_path}")
                import soundfile as sf
                import numpy as np
                
                audio, sr = sf.read(audio_path)
                duration = len(audio) / sr
                
                # Simple rule: detect energy changes to simulate speaker changes
                # This is a very basic fallback
                hop_length = int(sr * 0.5)  # 0.5 second windows
                energy = []
                
                for i in range(0, len(audio), hop_length):
                    window = audio[i:i+hop_length]
                    if len(window) > 0:
                        energy.append(np.sqrt(np.mean(window.astype(float)**2)))
                
                # Normalize energy
                energy = np.array(energy)
                if energy.max() > 0:
                    energy = energy / energy.max()
                
                # Find energy changes (simple speaker turn detection)
                speaker_changes = []
                for i in range(1, len(energy)):
                    if energy[i] < 0.1 and energy[i-1] > 0.3:  # Silence to speech transition
                        speaker_changes.append(i * 0.5)
                
                # Create segments based on energy changes
                if len(speaker_changes) == 0:
                    # No clear changes, single speaker
                    return [DiarSegment(start=0.0, end=duration, speaker="SPK_1")]
                
                # Add start and end
                boundaries = [0.0] + speaker_changes + [duration]
                segments = []
                
                for i in range(len(boundaries) - 1):
                    speaker = f"SPK_{(i % 3) + 1}"  # Rotate between 3 speakers max
                    segments.append(DiarSegment(
                        start=boundaries[i], 
                        end=boundaries[i+1], 
                        speaker=speaker
                    ))
                
                return segments
            else:
                # Unknown mode
                raise RuntimeError(f"Unknown diarization mode: {self._mode}")

            # Case 0: Sortformer output - List[List[str]] where inner list contains RTTM lines
            if isinstance(diar_output, list) and diar_output and isinstance(diar_output[0], list):
                # Extract RTTM lines from nested list
                rttm_lines = []
                for sublist in diar_output:
                    if isinstance(sublist, list):
                        rttm_lines.extend(sublist)
                    else:
                        rttm_lines.append(str(sublist))
                
                if rttm_lines:
                    logger.info(f"Parsing {len(rttm_lines)} RTTM lines from Sortformer output")
                    segments = self._parse_rttm(rttm_lines)
                    if segments:
                        return segments
            
            # Case 1: diar_output already list of dicts with start/end/speaker
            if isinstance(diar_output, list) and diar_output and isinstance(diar_output[0], dict):
                segments: List[DiarSegment] = []
                for seg in diar_output:
                    start = float(seg.get("start") or seg.get("st") or seg.get("t0") or 0.0)
                    end = float(seg.get("end") or seg.get("et") or seg.get("t1") or 0.0)
                    spk = str(seg.get("speaker") or seg.get("spk") or seg.get("label") or "SPK_1")
                    segments.append(DiarSegment(start=start, end=end, speaker=spk))
                return segments

            # Case 2: diar_output returns a dict with 'segments'
            if isinstance(diar_output, dict) and "segments" in diar_output:
                segments: List[DiarSegment] = []
                for seg in diar_output["segments"]:
                    start = float(seg.get("start", 0.0))
                    end = float(seg.get("end", 0.0))
                    spk = str(seg.get("speaker", "SPK_1"))
                    segments.append(DiarSegment(start=start, end=end, speaker=spk))
                return segments

            # Case 3: diar_output provides RTTM text or RTTM file path; try to parse
            # Normalize to RTTM lines
            rttm_lines: List[str] = []
            if isinstance(diar_output, str):
                # Could be a path or raw RTTM content
                if "\n" in diar_output and " SPEAKER " in diar_output:
                    rttm_lines = [line.strip() for line in diar_output.splitlines() if line.strip()]
                else:
                    # Treat as file path
                    try:
                        with open(diar_output, "r") as f:
                            rttm_lines = [line.strip() for line in f if line.strip()]
                    except Exception:
                        rttm_lines = []
            elif isinstance(diar_output, dict) and "rttm" in diar_output:
                rttm = diar_output["rttm"]
                if isinstance(rttm, str):
                    if "\n" in rttm and " SPEAKER " in rttm:
                        rttm_lines = [line.strip() for line in rttm.splitlines() if line.strip()]
                    else:
                        try:
                            with open(rttm, "r") as f:
                                rttm_lines = [line.strip() for line in f if line.strip()]
                        except Exception:
                            rttm_lines = []

            if rttm_lines:
                segments = self._parse_rttm(rttm_lines)
                if segments:
                    # Apply confidence-based filtering to improve diarization quality
                    filtered_segments = self._filter_segments_by_confidence(segments)
                    return filtered_segments

            # If nothing matched, fallback single segment (avoid hard failure in prod)
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            duration = (len(audio) / sr) if sr else 0.0
            return [DiarSegment(start=0.0, end=float(duration), speaker="SPK_1")]

        except Exception as e:
            # Final fallback to single-speaker if diarization failed at runtime
            logger.warning(f"Diarization fallback (single speaker) due to error: {e}")
            try:
                import soundfile as sf
                audio, sr = sf.read(audio_path)
                duration = (len(audio) / sr) if sr else 0.0
            except Exception:
                duration = 0.0
            return [DiarSegment(start=0.0, end=float(duration), speaker="SPK_1")]

    def _parse_rttm(self, lines: List[str]) -> List[DiarSegment]:
        segments: List[DiarSegment] = []
        
        for line in lines:
            try:
                if not line or not line.strip():
                    continue
                
                parts = line.split()
                
                # Check for simplified format: "start end speaker" (from Sortformer)
                if len(parts) == 3:
                    start = float(parts[0])
                    end = float(parts[1])
                    spk = parts[2]
                    # Normalize speaker labels: speaker_0 -> SPK_1, speaker_1 -> SPK_2
                    if spk.startswith("speaker_"):
                        try:
                            spk_num = int(spk.split("_")[1]) + 1
                            spk = f"SPK_{spk_num}"
                        except:
                            pass
                    elif not spk.startswith("SPK"):
                        # Only add prefix if it doesn't already have SPK (handles both SPK_ and SPK)
                        spk = f"SPK_{spk}"
                    segments.append(DiarSegment(start=start, end=end, speaker=spk))
                    continue
                
                # Standard RTTM format: SPEAKER <file-id> <1> <start> <dur> <NA> <NA> <speaker-id> <NA> <NA>
                if " SPEAKER " in line and len(parts) >= 8:
                    start = float(parts[3])
                    dur = float(parts[4])
                    end = start + dur
                    spk = parts[7]
                    # Normalize speaker labels
                    if not spk.startswith("SPK"):
                        # Only add prefix if it doesn't already have SPK (handles both SPK_ and SPK)
                        spk = f"SPK_{spk}"
                    segments.append(DiarSegment(start=start, end=end, speaker=spk))
                    
            except Exception as e:
                logger.debug(f"Failed to parse RTTM line '{line}': {e}")
                continue
        
        return segments

    def _filter_segments_by_confidence(self, segments: List[DiarSegment]) -> List[DiarSegment]:
        """
        Filter diarization segments based on confidence and temporal consistency.
        
        This method applies confidence-based filtering to reduce false speaker changes
        and improve temporal resolution without hardcoded word patterns.
        """
        if not segments:
            return segments
        
        filtered = []
        
        for i, seg in enumerate(segments):
            # Calculate segment duration
            duration = seg.end - seg.start
            
            # Filter out very short segments that are likely errors
            if duration < self.min_duration_on:
                logger.debug(f"Filtering out short segment: {seg.speaker} {seg.start:.2f}-{seg.end:.2f}s ({duration:.2f}s)")
                continue
            
            # Check for rapid speaker changes that might be errors
            if i > 0:
                prev_seg = filtered[-1] if filtered else segments[i-1]
                gap = seg.start - prev_seg.end
                
                # If there's a very small gap between same speaker segments, merge them
                if (gap < 0.5 and 
                    seg.speaker == prev_seg.speaker and
                    duration < 2.0):  # Short segment following same speaker
                    
                    logger.debug(f"Merging short segment with previous: {seg.speaker} {seg.start:.2f}-{seg.end:.2f}s")
                    # Extend previous segment
                    prev_seg.end = seg.end
                    continue
            
            # Check for suspicious rapid speaker changes
            if i > 0 and i < len(segments) - 1:
                prev_seg = segments[i-1]
                next_seg = segments[i+1]
                
                # If this is a short segment between two segments of the same speaker
                if (seg.speaker != prev_seg.speaker and 
                    seg.speaker != next_seg.speaker and
                    prev_seg.speaker == next_seg.speaker and
                    duration < 3.0):  # Short outlier segment
                    
                    gap_before = seg.start - prev_seg.end
                    gap_after = next_seg.start - seg.end
                    
                    # If gaps are small, this is likely a diarization error
                    if gap_before < 1.0 and gap_after < 1.0:
                        logger.debug(f"Filtering suspicious outlier: {seg.speaker} {seg.start:.2f}-{seg.end:.2f}s")
                        # Merge the three segments
                        prev_seg.end = next_seg.end
                        # Skip adding this segment and handle next_seg in next iteration
                        continue
            
            filtered.append(DiarSegment(start=seg.start, end=seg.end, speaker=seg.speaker))
        
        logger.info(f"Filtered {len(segments)} segments down to {len(filtered)} high-confidence segments")
        return filtered

    def _generate_chunks(self, duration: float) -> List[Dict]:
        """
        Generate chunk metadata for audio duration.
        Uses 24-minute chunks with 30-second overlap.
        """
        chunks = []
        
        # For short audio (≤24 minutes), create single chunk
        if duration <= self.chunk_duration:
            chunks.append({
                'start': 0.0,
                'end': duration,
                'duration': duration,
                'chunk_idx': 0,
                'total_chunks': 1,
                'overlap_start': 0.0,
                'overlap_end': 0.0
            })
            return chunks
        
        # For long audio, create multiple chunks with overlap
        effective_chunk_duration = self.chunk_duration - self.overlap_duration
        num_chunks = int(np.ceil((duration - self.overlap_duration) / effective_chunk_duration))
        
        for i in range(num_chunks):
            if i == 0:
                # First chunk: no overlap at start
                start = 0.0
                end = min(self.chunk_duration, duration)
                overlap_start = 0.0
                overlap_end = self.overlap_duration if end < duration else 0.0
            elif i == num_chunks - 1:
                # Last chunk: no overlap at end
                start = i * effective_chunk_duration
                end = duration
                overlap_start = self.overlap_duration
                overlap_end = 0.0
            else:
                # Middle chunks: overlap on both sides
                start = i * effective_chunk_duration
                end = min(start + self.chunk_duration, duration)
                overlap_start = self.overlap_duration
                overlap_end = self.overlap_duration if end < duration else 0.0
            
            chunks.append({
                'start': start,
                'end': end,
                'duration': end - start,
                'chunk_idx': i,
                'total_chunks': num_chunks,
                'overlap_start': overlap_start,
                'overlap_end': overlap_end
            })
        
        return chunks

    async def _diarize_chunk(
        self, 
        audio_path: str, 
        chunk: Dict, 
        max_speakers: int
    ) -> List[DiarSegment]:
        """
        Diarize a specific time range of audio using manifest-based approach.
        """
        # Create manifest entry for this chunk
        manifest_entry = {
            "audio_filepath": audio_path,
            "offset": chunk['start'],
            "duration": chunk['duration'],
            "label": "infer"
        }
        
        # Write temporary manifest file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(manifest_entry, f)
            f.write('\n')  # NeMo expects newline-delimited JSON
            manifest_path = f.name
        
        try:
            # Run diarization on this chunk
            loop = asyncio.get_event_loop()
            segments = await loop.run_in_executor(
                None,
                self._run_sync_manifest,
                manifest_path,
                max_speakers
            )
            
            # Adjust timestamps to global time (add chunk start offset)
            adjusted_segments = []
            for seg in segments:
                adjusted_segments.append(
                    DiarSegment(
                        start=seg.start + chunk['start'],
                        end=seg.end + chunk['start'],
                        speaker=seg.speaker
                    )
                )
            
            return adjusted_segments
            
        finally:
            # Clean up temporary manifest
            try:
                os.unlink(manifest_path)
            except Exception:
                pass

    def _run_sync_manifest(self, manifest_path: str, max_speakers: int) -> List[DiarSegment]:
        """
        Run diarization using a manifest file (for chunk processing).
        Extracts the audio segment and processes it directly.
        """
        try:
            import soundfile as sf
            
            # Read manifest to get audio segment info
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            audio_path = manifest['audio_filepath']
            offset = manifest.get('offset', 0.0)
            duration = manifest.get('duration')
            
            # Load audio segment
            audio, sr = sf.read(audio_path)
            start_sample = int(offset * sr)
            end_sample = int((offset + duration) * sr) if duration else len(audio)
            audio_segment = audio[start_sample:end_sample]
            
            # Save temporary audio file for this chunk
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio_segment, sr)
                tmp_path = tmp.name
            
            try:
                # Run diarization on temp file (timestamps will be relative to chunk start)
                if self._mode == 'sortformer':
                    logger.debug(f"Running Sortformer on chunk audio: {duration:.1f}s")
                    diar_output = self._model.diarize(audio=tmp_path, batch_size=1, verbose=False)
                    
                    # Parse output
                    if isinstance(diar_output, list) and len(diar_output) > 0:
                        if isinstance(diar_output[0], list):
                            # List of RTTM lines
                            segments = self._parse_rttm(diar_output[0])
                            logger.debug(f"Chunk diarization: {len(segments)} segments")
                            return segments
                    
                    logger.warning(f"Unexpected diarization output format: {type(diar_output)}")
                    return []
                else:
                    # Use generic _run_sync for other model types
                    return self._run_sync(tmp_path, max_speakers)
                    
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                        
        except Exception as e:
            logger.error(f"Manifest-based diarization failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def _ensure_speaker_model(self):
        """Load speaker embedding model for re-identification."""
        if self._speaker_model is not None:
            return
        
        async with self._lock:
            if self._speaker_model is not None:
                return
            
            try:
                import nemo.collections.asr as nemo_asr
                
                logger.info("Loading speaker embedding model: nvidia/speakerverification_en_titanet_large")
                self._speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                    'nvidia/speakerverification_en_titanet_large'
                )
                logger.info("Speaker embedding model loaded successfully")
                
            except Exception as e:
                logger.warning(f"Failed to load speaker embedding model: {e}")
                logger.warning("Speaker re-identification will use simplified matching")
                self._speaker_model = None

    async def _extract_overlap_embeddings(
        self,
        audio_path: str,
        chunk_results: List[Dict]
    ) -> Dict[Tuple[int, str], np.ndarray]:
        """
        Extract speaker embeddings from overlap regions between chunks.
        Returns dict mapping (chunk_idx, speaker_label) to embedding vector.
        """
        await self._ensure_speaker_model()
        
        if self._speaker_model is None:
            logger.warning("No speaker model available, skipping embedding extraction")
            return {}
        
        embeddings = {}
        
        for i, result in enumerate(chunk_results):
            chunk = result['chunk']
            segments = result['segments']
            
            # Get unique speakers in this chunk
            speakers = set(seg.speaker for seg in segments)
            
            for speaker_label in speakers:
                # Extract embedding from all segments of this speaker in this chunk
                # (not just overlap - more robust for matching)
                speaker_segments = [
                    seg for seg in segments
                    if seg.speaker == speaker_label
                ]
                
                if speaker_segments:
                    # Extract embedding for this speaker
                    embedding = await self._get_speaker_embedding(
                        audio_path,
                        speaker_segments
                    )
                    if embedding is not None:
                        embeddings[(i, speaker_label)] = embedding
                        logger.debug(f"Extracted embedding for chunk {i} speaker {speaker_label}")
        
        return embeddings

    async def _get_speaker_embedding(
        self,
        audio_path: str,
        segments: List[DiarSegment]
    ) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio segments.
        """
        if self._speaker_model is None:
            return None
        
        try:
            import soundfile as sf
            
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Extract audio for these segments
            speaker_audio = []
            for seg in segments:
                start_sample = int(seg.start * sr)
                end_sample = int(seg.end * sr)
                speaker_audio.append(audio[start_sample:end_sample])
            
            # Concatenate all segments for this speaker
            if not speaker_audio:
                return None
            
            speaker_audio = np.concatenate(speaker_audio)
            
            # Save temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, speaker_audio, sr)
                tmp_path = tmp.name
            
            try:
                # Extract embedding
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None,
                    self._extract_embedding_sync,
                    tmp_path
                )
                return embedding
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug(f"Failed to extract speaker embedding: {e}")
            return None

    def _extract_embedding_sync(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract speaker embedding synchronously."""
        try:
            # Get embedding from speaker model
            embedding = self._speaker_model.get_embedding(audio_path)
            
            if isinstance(embedding, np.ndarray):
                return embedding
            elif hasattr(embedding, 'cpu'):
                # PyTorch tensor
                return embedding.cpu().numpy()
            else:
                return np.array(embedding)
                
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
            return None

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Check if embeddings are valid
            if emb1 is None or emb2 is None:
                logger.warning("One or both embeddings are None")
                return 0.0
            
            # Flatten to 1D if needed
            emb1_flat = emb1.flatten()
            emb2_flat = emb2.flatten()
            
            if emb1_flat.size == 0 or emb2_flat.size == 0:
                logger.warning(f"Empty embedding: emb1.size={emb1_flat.size}, emb2.size={emb2_flat.size}")
                return 0.0
            
            # Normalize embeddings
            norm1 = np.linalg.norm(emb1_flat)
            norm2 = np.linalg.norm(emb2_flat)
            
            if norm1 == 0 or norm2 == 0:
                logger.warning(f"Zero norm: norm1={norm1}, norm2={norm2}")
                return 0.0
            
            emb1_norm = emb1_flat / norm1
            emb2_norm = emb2_flat / norm2
            
            # Cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0

    async def _merge_and_reidentify(
        self,
        chunk_results: List[Dict],
        embeddings: Dict[Tuple[int, str], np.ndarray]
    ) -> List[DiarSegment]:
        """
        Merge segments across chunks and assign consistent speaker IDs.
        Uses embedding-based similarity matching to identify the same speaker across chunks.
        """
        if not chunk_results:
            return []
        
        # Initialize speaker ID mapping
        # Format: {(chunk_idx, local_speaker): global_speaker}
        speaker_mapping = {}
        next_global_id = 1
        
        # Track global speaker embeddings for matching
        # Format: {global_speaker: embedding}
        global_speaker_embeddings = {}
        
        # First chunk: initialize global speakers
        first_chunk_speakers = set(seg.speaker for seg in chunk_results[0]['segments'])
        for local_speaker in sorted(first_chunk_speakers):
            global_speaker = f"SPK_{next_global_id}"
            speaker_mapping[(0, local_speaker)] = global_speaker
            
            # Store embedding if available
            emb = embeddings.get((0, local_speaker))
            if emb is not None:
                global_speaker_embeddings[global_speaker] = emb
            
            next_global_id += 1
        
        logger.info(f"Chunk 0: Initialized {len(first_chunk_speakers)} speakers")
        
        # Process subsequent chunks using embedding similarity
        for i in range(1, len(chunk_results)):
            current_speakers = set(seg.speaker for seg in chunk_results[i]['segments'])
            matched_count = 0
            
            for curr_speaker in sorted(current_speakers):
                curr_emb = embeddings.get((i, curr_speaker))
                
                best_match = None
                best_similarity = 0.0
                
                if curr_emb is not None:
                    # Compare with all existing global speakers using embeddings
                    for global_speaker, global_emb in global_speaker_embeddings.items():
                        similarity = self._compute_similarity(curr_emb, global_emb)
                        logger.debug(f"Chunk {i} {curr_speaker} vs {global_speaker}: similarity={similarity:.3f}")
                        
                        if similarity > best_similarity and similarity >= self.similarity_threshold:
                            best_similarity = similarity
                            best_match = global_speaker
                
                if best_match:
                    # Matched with existing speaker
                    speaker_mapping[(i, curr_speaker)] = best_match
                    matched_count += 1
                    logger.debug(f"Chunk {i} {curr_speaker} → {best_match} (similarity: {best_similarity:.3f})")
                else:
                    # New speaker
                    new_global_speaker = f"SPK_{next_global_id}"
                    speaker_mapping[(i, curr_speaker)] = new_global_speaker
                    
                    # Store embedding for future matching
                    if curr_emb is not None:
                        global_speaker_embeddings[new_global_speaker] = curr_emb
                    
                    logger.debug(f"Chunk {i} {curr_speaker} → {new_global_speaker} (new speaker)")
                    next_global_id += 1
            
            logger.info(f"Chunk {i}: {matched_count} speakers matched, {len(current_speakers) - matched_count} new speakers")
        
        # Apply mapping to all segments
        merged_segments = []
        for i, result in enumerate(chunk_results):
            for seg in result['segments']:
                global_speaker = speaker_mapping.get((i, seg.speaker), seg.speaker)
                merged_segments.append(
                    DiarSegment(
                        start=seg.start,
                        end=seg.end,
                        speaker=global_speaker
                    )
                )
        
        # Sort by start time
        merged_segments.sort(key=lambda s: s.start)
        
        # Remove duplicate segments in overlap regions
        merged_segments = self._deduplicate_overlaps(merged_segments)
        
        return merged_segments

    def _deduplicate_overlaps(
        self,
        segments: List[DiarSegment],
        overlap_tolerance: float = 0.5
    ) -> List[DiarSegment]:
        """
        Remove duplicate segments in overlap regions.
        Keeps the segment with better coverage.
        """
        if not segments:
            return []
        
        deduplicated = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # Check if next segment overlaps significantly
            if i + 1 < len(segments):
                next_seg = segments[i + 1]
                
                # Calculate overlap
                overlap_start = max(current.start, next_seg.start)
                overlap_end = min(current.end, next_seg.end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # If significant overlap and same speaker, merge
                if (overlap_duration > overlap_tolerance and 
                    current.speaker == next_seg.speaker):
                    # Merge segments
                    merged = DiarSegment(
                        start=min(current.start, next_seg.start),
                        end=max(current.end, next_seg.end),
                        speaker=current.speaker
                    )
                    deduplicated.append(merged)
                    i += 2  # Skip both segments
                    continue
            
            deduplicated.append(current)
            i += 1
        
        return deduplicated


async def store_diar_segments(redis_client, job_id: str, segments: List[DiarSegment], num_speakers: int):
    """Store diarization segments in Redis under a standardized key."""
    import json
    key = f"diar:{job_id}:segments"
    data = {
        "numSpeakers": num_speakers,
        "segments": [
            {"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments
        ],
    }
    await redis_client.set(key, json.dumps(data).encode("utf-8"))


async def load_diar_segments(redis_client, job_id: str) -> Optional[Dict]:
    import json
    key = f"diar:{job_id}:segments"
    raw = await redis_client.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


