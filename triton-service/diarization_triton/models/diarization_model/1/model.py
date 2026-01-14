"""
Triton Inference Server - Speaker Diarization with NVIDIA MSDD
WORKING VERSION: Correct MSDD parameter names and feature extraction

Model: NVIDIA MSDD for telephonic diarization
Optimized for: 2-speaker CSR phone calls
"""

import os
import json
import uuid
import logging
import warnings
from typing import List, Dict, Any

import numpy as np
import triton_python_backend_utils as pb_utils

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


class TritonPythonModel:

    def initialize(self, args):
        """Initialize with proper NeMo configuration"""
        import torch
        
        self.model_config = json.loads(args["model_config"])
        self.model_instance_name = args["model_instance_name"]
        self.device_id = int(args["model_instance_device_id"])
        self.device = f"cuda:{self.device_id}"

        print(f"[MSDD Diarizer] Initializing on {self.device}")

        # CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.set_grad_enabled(False)

        self.sample_rate = 16000
        self.num_speakers = 2

        # Work directory
        self.work_dir = f"/tmp/msdd_{self.model_instance_name}"
        os.makedirs(self.work_dir, exist_ok=True)

        # Initialize models
        self._init_models()
        
        print("[MSDD Diarizer] ✓ Ready for telephonic diarization")

    def _init_models(self):
        """Initialize VAD, Speaker Embedding, and MSDD models"""
        from nemo.collections.asr.models import EncDecClassificationModel
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        from nemo.collections.asr.models import EncDecDiarLabelModel
        
        print("[MSDD Diarizer] Loading models...")
        
        # VAD
        self.vad_model = EncDecClassificationModel.from_pretrained(
            'vad_multilingual_marblenet'
        ).to(self.device).eval()
        print("  ✓ VAD model loaded")
        
        # TitaNet
        self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(
            'titanet_large'
        ).to(self.device).eval()
        print("  ✓ TitaNet speaker model loaded")
        
        # MSDD
        self.msdd_model = EncDecDiarLabelModel.from_pretrained(
            'diar_msdd_telephonic'
        ).to(self.device).eval()
        print("  ✓ MSDD model loaded")

    # -------------------- Audio Preprocessing --------------------

    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess audio to 16kHz mono"""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sample_rate != self.sample_rate:
            from scipy.signal import resample
            num_samples = int(len(audio) * self.sample_rate / sample_rate)
            audio = resample(audio, num_samples)
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        audio = np.clip(audio * 0.95, -1.0, 1.0)

        return audio.astype(np.float32)

    # -------------------- VAD --------------------

    def run_vad(self, audio: np.ndarray) -> List[tuple]:
        """Run Voice Activity Detection"""
        import torch
        
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        audio_length = torch.tensor([audio_tensor.shape[1]], device=self.device)
        
        with torch.no_grad():
            logits = self.vad_model(
                input_signal=audio_tensor,
                input_signal_length=audio_length
            )
        
        if isinstance(logits, tuple):
            logits = logits[0]
        
        probs = torch.softmax(logits, dim=-1)
        
        if probs.dim() == 3:
            speech_probs = probs[0, :, 1].cpu().numpy()
            segments = self._vad_to_segments(speech_probs)
        elif probs.dim() == 2:
            speech_prob = probs[0, 1].item()
            if speech_prob > 0.5:
                duration = len(audio) / self.sample_rate
                segments = [(0.0, duration)]
            else:
                segments = []
        else:
            duration = len(audio) / self.sample_rate
            segments = [(0.0, duration)]
        
        return segments

    def _vad_to_segments(
        self, 
        speech_probs: np.ndarray,
        threshold: float = 0.5,
        min_duration: float = 0.2,
        frame_duration: float = 0.02
    ) -> List[tuple]:
        """Convert VAD probabilities to speech segments"""
        
        speech_frames = speech_probs > threshold
        
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                in_speech = True
                start_frame = i
            elif not is_speech and in_speech:
                in_speech = False
                start_time = start_frame * frame_duration
                end_time = i * frame_duration
                if end_time - start_time >= min_duration:
                    segments.append((start_time, end_time))
        
        if in_speech:
            start_time = start_frame * frame_duration
            end_time = len(speech_frames) * frame_duration
            if end_time - start_time >= min_duration:
                segments.append((start_time, end_time))
        
        return segments

    # -------------------- Speaker Embeddings --------------------

    def extract_embeddings(
        self, 
        audio: np.ndarray,
        segments: List[tuple],
        window_length: float = 1.5,
        shift_length: float = 0.75
    ) -> tuple:
        """Extract speaker embeddings"""
        import torch
        
        embeddings = []
        embedding_times = []
        
        for seg_start, seg_end in segments:
            current_time = seg_start
            
            while current_time + window_length <= seg_end:
                start_sample = int(current_time * self.sample_rate)
                end_sample = int((current_time + window_length) * self.sample_rate)
                
                if end_sample > len(audio):
                    break
                
                chunk = audio[start_sample:end_sample]
                chunk_tensor = torch.from_numpy(chunk).float().to(self.device)
                if chunk_tensor.dim() == 1:
                    chunk_tensor = chunk_tensor.unsqueeze(0)
                
                chunk_length = torch.tensor([chunk_tensor.shape[1]], device=self.device)
                
                with torch.no_grad():
                    _, emb = self.speaker_model(
                        input_signal=chunk_tensor,
                        input_signal_length=chunk_length
                    )
                
                embeddings.append(emb.cpu().numpy().flatten())
                embedding_times.append((current_time, current_time + window_length))
                
                current_time += shift_length
        
        if len(embeddings) == 0:
            return np.array([]), []
        
        return np.vstack(embeddings), embedding_times

    # -------------------- Clustering --------------------

    def cluster_embeddings(self, embeddings: np.ndarray, num_speakers: int = 2) -> np.ndarray:
        """Cluster embeddings"""
        from sklearn.cluster import SpectralClustering
        from sklearn.preprocessing import normalize
        
        if len(embeddings) < num_speakers:
            return np.zeros(len(embeddings), dtype=int)
        
        embeddings_norm = normalize(embeddings)
        affinity = np.dot(embeddings_norm, embeddings_norm.T)
        affinity = (affinity + 1) / 2
        
        try:
            clustering = SpectralClustering(
                n_clusters=num_speakers,
                affinity='precomputed',
                n_init=10,
                random_state=42
            )
            labels = clustering.fit_predict(affinity)
        except:
            labels = np.zeros(len(embeddings), dtype=int)
            labels[len(embeddings)//2:] = 1
        
        return labels

    # -------------------- MSDD Refinement --------------------

    def run_msdd(
        self,
        audio: np.ndarray,
        embeddings: np.ndarray,
        embedding_times: List[tuple],
        initial_labels: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        MSDD refinement - pass audio through encoder to get features
        """
        import torch
        
        if len(embeddings) == 0:
            return []
        
        num_speakers = self.num_speakers
        seq_len = len(embeddings)
        
        print(f"[DEBUG] MSDD input: seq_len={seq_len}, num_speakers={num_speakers}")
        
        # Convert audio to tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long, device=self.device)
        
        # Get audio features from MSDD encoder
        with torch.no_grad():
            try:
                # Extract features using the model's preprocessor/encoder
                processed_signal, processed_length = self.msdd_model.preprocessor(
                    input_signal=audio_tensor,
                    length=audio_length
                )
                
                # Get encoder features
                encoded, encoded_len = self.msdd_model.encoder(
                    audio_signal=processed_signal,
                    length=processed_length
                )
            except Exception as e:
                print(f"[WARNING] Failed to extract MSDD features: {e}")
                return self._labels_to_segments(embedding_times, initial_labels)
        
        features = encoded  # Shape: (batch, time, feature_dim)
        feature_length = encoded_len
        
        print(f"[DEBUG] Encoded features shape: {features.shape}")
        
        # Create timestamps for the encoded features
        # Map embedding timestamps to encoder frame indices
        total_duration = len(audio) / self.sample_rate
        feature_frames = features.shape[1]
        frame_duration = total_duration / feature_frames
        
        # Create timestamps aligned with encoder frames
        timestamps = np.zeros((1, feature_frames, 2), dtype=np.float32)
        for i in range(feature_frames):
            timestamps[0, i, 0] = i * frame_duration
            timestamps[0, i, 1] = (i + 1) * frame_duration
        ms_seg_timestamps = torch.from_numpy(timestamps).to(self.device)
        
        # Scale configuration
        num_scales = 5
        scale_mapping = [[0.5, 0.25], [0.75, 0.375], [1.0, 0.5], [1.25, 0.625], [1.5, 0.75]]
        
        # Segment counts per scale
        seg_counts = np.full((1, num_scales), feature_frames, dtype=np.int64)
        ms_seg_counts = torch.from_numpy(seg_counts).to(self.device)
        
        # Map initial clustering labels to encoder frames
        clus_labels = torch.zeros(1, feature_frames, dtype=torch.long, device=self.device)
        for i, (start, end) in enumerate(embedding_times):
            label = initial_labels[i]
            # Find corresponding frame indices
            start_frame = int(start / frame_duration)
            end_frame = int(end / frame_duration)
            end_frame = min(end_frame, feature_frames)
            clus_labels[0, start_frame:end_frame] = label
        
        # Create target labels
        targets = torch.zeros(1, feature_frames, num_speakers, device=self.device)
        for i in range(feature_frames):
            label = clus_labels[0, i].item()
            targets[0, i, label] = 1.0
        
        print(f"[DEBUG] MSDD tensor shapes:")
        print(f"  features: {features.shape}")
        print(f"  feature_length: {feature_length}")
        print(f"  ms_seg_timestamps: {ms_seg_timestamps.shape}")
        print(f"  ms_seg_counts: {ms_seg_counts.shape}")
        print(f"  clus_labels: {clus_labels.shape}")
        print(f"  targets: {targets.shape}")
        
        # Run MSDD
        with torch.no_grad():
            try:
                output = self.msdd_model(
                    features=features,
                    feature_length=feature_length,
                    ms_seg_timestamps=ms_seg_timestamps,
                    ms_seg_counts=ms_seg_counts,
                    clus_label_index=clus_labels,
                    scale_mapping=scale_mapping,
                    targets=targets
                )
                
                print(f"[DEBUG] MSDD output type: {type(output)}")
                
                if isinstance(output, tuple):
                    probs = output[0]
                else:
                    probs = output
                
                print(f"[DEBUG] MSDD probs shape: {probs.shape}")
                
                # Extract probabilities
                if probs.dim() == 3:
                    probs = probs[0].cpu().numpy()
                elif probs.dim() == 2:
                    probs = probs.cpu().numpy()
                else:
                    print(f"[ERROR] Unexpected probs shape: {probs.shape}")
                    return self._labels_to_segments(embedding_times, initial_labels)
                
                print(f"[DEBUG] Final probs shape: {probs.shape}")
                print(f"[SUCCESS] ✓ MSDD refinement completed successfully")
                
            except Exception as e:
                print(f"[WARNING] MSDD forward failed: {e}")
                import traceback
                print(traceback.format_exc())
                return self._labels_to_segments(embedding_times, initial_labels)
        
        # Convert encoder frame probabilities back to segments
        segments = self._msdd_to_segments_from_frames(probs, frame_duration)
        
        return segments

    def _msdd_to_segments_from_frames(
        self,
        probs: np.ndarray,
        frame_duration: float,
        threshold: float = 0.5,
        min_duration: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Convert MSDD frame-level probabilities to segments"""
        
        if probs.ndim != 2:
            print(f"[ERROR] Expected 2D probs, got {probs.shape}")
            return []
        
        speaker_ids = np.argmax(probs, axis=1)
        speaker_scores = probs[np.arange(len(probs)), speaker_ids]
        speaker_ids[speaker_scores < threshold] = -1
        
        segments = []
        current_speaker = speaker_ids[0]
        seg_start_frame = 0
        
        for i in range(1, len(speaker_ids)):
            if speaker_ids[i] != current_speaker:
                if current_speaker != -1:
                    start_time = seg_start_frame * frame_duration
                    end_time = i * frame_duration
                    
                    if end_time - start_time >= min_duration:
                        segments.append({
                            "speaker": f"speaker_{current_speaker}",
                            "start": round(start_time, 3),
                            "end": round(end_time, 3),
                            "duration": round(end_time - start_time, 3)
                        })
                
                seg_start_frame = i
                current_speaker = speaker_ids[i]
        
        if current_speaker != -1:
            start_time = seg_start_frame * frame_duration
            end_time = len(speaker_ids) * frame_duration
            if end_time - start_time >= min_duration:
                segments.append({
                    "speaker": f"speaker_{current_speaker}",
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "duration": round(end_time - start_time, 3)
                })
        
        return segments

    def _labels_to_segments(
        self,
        times: List[tuple],
        labels: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Convert labels to segments (fallback)"""
        
        segments = []
        current_label = labels[0]
        seg_start_idx = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                start_time = times[seg_start_idx][0]
                end_time = times[i-1][1]
                
                segments.append({
                    "speaker": f"speaker_{current_label}",
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "duration": round(end_time - start_time, 3)
                })
                
                seg_start_idx = i
                current_label = labels[i]
        
        start_time = times[seg_start_idx][0]
        end_time = times[-1][1]
        segments.append({
            "speaker": f"speaker_{current_label}",
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "duration": round(end_time - start_time, 3)
        })
        
        return segments

    # -------------------- Post-processing --------------------

    def merge_segments(
        self,
        segments: List[Dict[str, Any]],
        gap_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Merge adjacent same-speaker segments"""
        if not segments:
            return []
        
        segments = sorted(segments, key=lambda x: x["start"])
        merged = [segments[0].copy()]
        
        for seg in segments[1:]:
            prev = merged[-1]
            
            if (seg["speaker"] == prev["speaker"] and 
                seg["start"] - prev["end"] <= gap_threshold):
                prev["end"] = seg["end"]
                prev["duration"] = round(prev["end"] - prev["start"], 3)
            else:
                merged.append(seg.copy())
        
        return merged

    # -------------------- Main Processing --------------------

    def process_audio(self, audio: np.ndarray, request_id: str) -> Dict[str, Any]:
        """Main diarization pipeline"""
        import time
        
        duration = len(audio) / self.sample_rate
        
        print(f"\n{'='*80}")
        print(f"[{request_id}] Processing {duration:.2f}s audio")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            print(f"[{request_id}] Step 1: Running VAD...")
            vad_segments = self.run_vad(audio)
            print(f"[{request_id}]   ✓ Found {len(vad_segments)} speech segments")
            
            if len(vad_segments) == 0:
                return {
                    "segments": [],
                    "num_speakers": 0,
                    "total_segments": 0,
                    "duration": round(duration, 3),
                    "inference_time": round(time.time() - start_time, 3),
                    "status": "success"
                }
            
            print(f"[{request_id}] Step 2: Extracting speaker embeddings...")
            embeddings, embedding_times = self.extract_embeddings(audio, vad_segments)
            print(f"[{request_id}]   ✓ Extracted {len(embeddings)} embeddings")
            
            if len(embeddings) == 0:
                return {
                    "segments": [],
                    "num_speakers": 0,
                    "total_segments": 0,
                    "duration": round(duration, 3),
                    "inference_time": round(time.time() - start_time, 3),
                    "status": "success"
                }
            
            print(f"[{request_id}] Step 3: Clustering speakers...")
            initial_labels = self.cluster_embeddings(embeddings, self.num_speakers)
            unique_labels = len(set(initial_labels))
            print(f"[{request_id}]   ✓ Found {unique_labels} speakers in clustering")
            
            print(f"[{request_id}] Step 4: Running MSDD refinement...")
            segments = self.run_msdd(audio, embeddings, embedding_times, initial_labels)
            print(f"[{request_id}]   ✓ MSDD generated {len(segments)} segments")
            
            print(f"[{request_id}] Step 5: Merging adjacent same-speaker segments...")
            segments = self.merge_segments(segments)
            
            inference_time = time.time() - start_time
            final_speakers = set(s["speaker"] for s in segments) if segments else set()
            
            print(f"[{request_id}] ✓ Complete: {len(segments)} segments, {len(final_speakers)} speakers ({inference_time:.2f}s)")
            print(f"{'='*80}\n")
            
            return {
                "segments": segments,
                "num_speakers": len(final_speakers),
                "total_segments": len(segments),
                "duration": round(duration, 3),
                "inference_time": round(inference_time, 3),
                "status": "success"
            }
            
        except Exception as e:
            import traceback
            print(f"[{request_id}] ✗ Error: {str(e)}")
            print(traceback.format_exc())
            raise

    # -------------------- Triton Interface --------------------

    def execute(self, requests):
        """Triton execution endpoint"""
        responses = []

        for request in requests:
            try:
                audio_input = pb_utils.get_input_tensor_by_name(request, "audio_input")
                sr_input = pb_utils.get_input_tensor_by_name(request, "sample_rate")

                audio = audio_input.as_numpy()
                if audio.ndim == 2:
                    audio = audio[0]

                sample_rate = int(sr_input.as_numpy().flat[0])
                request_id = f"{self.model_instance_name}_{uuid.uuid4().hex[:8]}"

                audio = self.preprocess_audio(audio, sample_rate)
                result = self.process_audio(audio, request_id)
                
                print(f"diar_result: {json.dumps(result)}\n", flush=True)

                out = pb_utils.Tensor(
                    "diarization_output",
                    np.array([json.dumps(result)], dtype=object),
                )

                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[out])
                )

            except Exception as e:
                import traceback
                print(f"[ERROR] {str(e)}")
                print(traceback.format_exc())
                
                err = {
                    "status": "error",
                    "error": str(e),
                    "segments": [],
                    "num_speakers": 0,
                }
                
                out = pb_utils.Tensor(
                    "diarization_output",
                    np.array([json.dumps(err)], dtype=object),
                )
                
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[out],
                        error=pb_utils.TritonError(str(e)),
                    )
                )

        return responses

    def finalize(self):
        """Cleanup"""
        import shutil
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)
        print(f"[MSDD Diarizer] Shutdown: {self.model_instance_name}")