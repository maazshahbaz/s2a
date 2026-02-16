import asyncio
import json
import os
import soundfile as sf
from typing import Tuple, Dict, List
from .config_loader import config

from .audio_chunking import AudioChunking
from .transcription_client import AsyncTranscriptionService
from .diarization_client import AsyncDiarizationClient
from .analysis_client import AsyncAnalysis
from .scoring_client import AsyncCSRScoringClient
from .fraud_client import AsyncFraudDetectionClient
from .email_client import AsyncFollowUpEmailClient
from .transcript_merger import TranscriptMerger


class Pipeline:
    """
    Audio processing pipeline with GLOBAL speaker consistency.
    
    Key improvement: Aligns speaker labels across chunks using embedding similarity.
    
    Steps:
    1. Chunk audio to 5-minute intervals at 16 KHz
    2a. Run diarization on FULL audio (for global reference)
    2b. Run ASR on chunks (parallel)
    3. Align chunk-based transcription with global diarization
    4. Run AI analysis and CSR agent scoring in parallel
    """
    
    def __init__(self,
        diarization_url: str = None,
        transcription_url: str = None,
        csr_scoring_url: str = None,
        fraud_detection_url: str = None):
        """
        Initialize pipeline with service URLs from config or provided values.

        Args:
            diarization_url: Override diarization service URL (default: from config)
            transcription_url: Override transcription service URL (default: from config)
            csr_scoring_url: Override CSR scoring service URL (default: from config)
            fraud_detection_url: Override fraud detection service URL (default: from config)
        """
        # Load service configurations
        diar_config = config.get_service_config('diarization')
        trans_config = config.get_service_config('transcription')
        scoring_config = config.get_service_config('csr_scoring')
        fraud_config = config.get_service_config('fraud_detection')

        self.chunking = AudioChunking()
        self.transcription = AsyncTranscriptionService(
            url=transcription_url or trans_config.get('url')
        )
        self.diarization = AsyncDiarizationClient(
            url=diarization_url or diar_config.get('url')
        )
        self.analysis = AsyncAnalysis()
        self.csr_scoring = AsyncCSRScoringClient(
            url=csr_scoring_url or scoring_config.get('url')
        )
        self.fraud_detection = AsyncFraudDetectionClient(
            url=fraud_detection_url or fraud_config.get('url')
        )
        self.merger = TranscriptMerger()
    
    async def _delete_chunks_async(self, chunk_paths: List[str]):
        """Delete all chunk files asynchronously."""
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, os.remove, path) for path in chunk_paths]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        audio_data, sample_rate = sf.read(audio_path)
        return len(audio_data) / sample_rate
    
    async def _run_global_diarization(self, audio_path: str, request_id: str) -> Dict:
        """
        Step 2a: Run diarization on FULL audio file for global speaker consistency.
        
        This provides a reference for consistent speaker labels across all chunks.
        """
        await self.diarization.connect()
        
        global_diar_result = await self.diarization.diarize(audio_path, request_id)
        
        return global_diar_result
    
    async def _run_transcription_on_chunks(self, chunk_paths: List[str], request_id: str) -> List[Dict]:
        """Step 2b: Run transcription on all chunks in parallel."""
        chunk_basenames = [os.path.basename(path) for path in chunk_paths]
        
        transcription_tasks = [
            self.transcription.transcribe_async(basename, f"{request_id}_chunk_{i}")
            for i, basename in enumerate(chunk_basenames)
        ]
        
        transcription_results = await asyncio.gather(*transcription_tasks)
        return transcription_results
    
    def _align_segments_to_global(
        self,
        global_segments: List[Dict],
        chunk_timings: List[Tuple[float, float]]
    ) -> List[List[Dict]]:
        """
        Split global diarization segments into per-chunk segments.
        
        Args:
            global_segments: Segments from full audio diarization
            chunk_timings: [(start, end), ...] for each chunk
            
        Returns:
            List of segment lists, one per chunk
        """
        chunk_segments = []
        
        for chunk_start, chunk_end in chunk_timings:
            chunk_segs = []
            
            for seg in global_segments:
                seg_start = seg['start']
                seg_end = seg['end']
                
                # Check if segment overlaps with this chunk
                if seg_end <= chunk_start or seg_start >= chunk_end:
                    continue  # No overlap
                
                # Clip segment to chunk boundaries and adjust to chunk-local time
                local_start = max(0, seg_start - chunk_start)
                local_end = min(chunk_end - chunk_start, seg_end - chunk_start)
                
                chunk_segs.append({
                    'speaker': seg['speaker'],
                    'start': local_start,
                    'end': local_end,
                    'duration': local_end - local_start
                })
            
            chunk_segments.append(chunk_segs)
        
        return chunk_segments
    
    async def _run_analysis_and_scoring(
        self,
        labeled_transcription: str,
        request_id: str
    ) -> Dict:
        """
        Run AI analysis, CSR agent scoring, and fraud detection in parallel.

        Args:
            labeled_transcription: The labeled transcript text
            request_id: Unique request identifier

        Returns:
            Analysis dict with 'agent_scoring' and 'fraud_detection' fields embedded
        """
        await self.csr_scoring.connect()

        # Run all three in parallel
        analysis_task = self.analysis.analyze_call_async(labeled_transcription, request_id)
        scoring_task = self.csr_scoring.score_transcript(
            transcript=labeled_transcription,
            request_id=f"{request_id}"
        )
        fraud_task = self.fraud_detection.detect_fraud(
            transcript=labeled_transcription,
            request_id=f"{request_id}_fraud"
        )

        analysis_result, scoring_result, fraud_result = await asyncio.gather(
            analysis_task,
            scoring_task,
            fraud_task
        )

        # Parse analysis_result if it's a JSON string
        if isinstance(analysis_result, str):
            try:
                analysis_result = json.loads(analysis_result)
            except json.JSONDecodeError:
                print(f"[Pipeline] Failed to parse analysis result as JSON")
                analysis_result = {"error": "Failed to parse analysis", "raw": analysis_result}

        # Extract agent_scoring from nested structure if present
        if isinstance(scoring_result, dict) and 'agent_scoring' in scoring_result:
            agent_scoring = scoring_result['agent_scoring']
        else:
            agent_scoring = scoring_result

        # Extract fraud_detection from nested structure if present
        if isinstance(fraud_result, dict) and 'fraud_detection' in fraud_result:
            fraud_detection = fraud_result['fraud_detection']
        else:
            fraud_detection = fraud_result

        # Add agent_scoring to the analysis result
        analysis_result['agent_scoring'] = agent_scoring

        # Merge fraud_detection into ai_analysis for backward compatibility
        if 'analysis' in analysis_result and 'ai_analysis' in analysis_result.get('analysis', {}):
            analysis_result['analysis']['ai_analysis']['fraud_detection'] = fraud_detection
        elif 'ai_analysis' in analysis_result:
            analysis_result['ai_analysis']['fraud_detection'] = fraud_detection
        else:
            analysis_result['fraud_detection'] = fraud_detection

        return analysis_result
    
    async def run_pipeline(
        self,
        audio_path: str,
        request_id: str
    ) -> Tuple[str, str, Dict, Dict]:
        """
        Execute the complete pipeline with global speaker consistency.
        
        Args:
            audio_path: Path to input audio file
            request_id: Unique identifier for this request
            
        Returns:
            Tuple of (raw_transcription, labeled_transcription, analysis_with_scoring, metadata)
        """
        
        # Step 1: Chunk audio to 5-minute intervals
        chunk_paths, chunk_timings = await self.chunking.create_chunks_async(audio_path)
        
        # Step 2: Run global diarization and chunk transcription in parallel
        global_diar_task = self._run_global_diarization(audio_path, request_id)
        transcription_task = self._run_transcription_on_chunks(chunk_paths, request_id)
        
        global_diar_result, transcription_results = await asyncio.gather(
            global_diar_task,
            transcription_task
        )
        
        # Extract global segments
        global_segments = global_diar_result.get('segments', [])
        print(f"[Pipeline] Global diarization: {len(global_segments)} segments, "
              f"{global_diar_result.get('num_speakers', 0)} speakers")
        
        # Step 3: Align global segments to chunk boundaries
        chunk_diarization = self._align_segments_to_global(global_segments, chunk_timings)
        
        # Step 4: Merge transcriptions with globally-consistent diarization
        raw_transcription, labeled_transcription = await self.merger.merge_transcriptions(
            request_id,
            transcription_results,
            [{'segments': segs} for segs in chunk_diarization],
            chunk_timings
        )
        
        # Step 5: Run AI analysis, CSR scoring, and cleanup chunks in parallel
        if not labeled_transcription:
            # If transcription is empty or None, skip analysis/scoring and just cleanup
            combined_analysis = None
            await self._delete_chunks_async(chunk_paths)
        else:
            analysis_scoring_task = self._run_analysis_and_scoring(
                labeled_transcription,
                request_id
            )
            cleanup_task = self._delete_chunks_async(chunk_paths)
            
            combined_analysis, _ = await asyncio.gather(
                analysis_scoring_task,
                cleanup_task
            )

            # Step 6: Generate follow-up email
            followup_result = await self._run_followup_email(
                labeled_transcription,
                combined_analysis,
                request_id
            )
            followup_result = followup_result['output']["follow_up_email"]
            if 'analysis' in combined_analysis and 'ai_analysis' in combined_analysis.get('analysis', {}):
                combined_analysis['analysis']['ai_analysis']['follow_up_email'] = followup_result

            elif 'ai_analysis' in combined_analysis:
                combined_analysis['ai_analysis']['follow_up_email'] = followup_result

            else:
                combined_analysis['follow_up_email'] = followup_result  
        
        
        # Close connections
        await self.diarization.close()
        await self.csr_scoring.close()
        await self.fraud_detection.close()
        
        # Prepare metadata
        metadata = {
            'chunk_count': len(chunk_paths),
            'chunk_timings': chunk_timings,
            'num_speakers': global_diar_result.get('num_speakers', 0),
            'total_segments': len(global_segments)
        }
        
        return raw_transcription, labeled_transcription, combined_analysis, metadata