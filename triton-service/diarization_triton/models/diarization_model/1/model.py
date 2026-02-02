"""
Triton Inference Server - Speaker Diarization with NVIDIA MSDD

Model: NVIDIA MSDD for telephonic diarization
Optimized for: 2-3 speaker conversations with IVR/DTMF filtering
"""

import os
import json
import uuid
import logging
import warnings
from typing import List, Dict, Any, Tuple

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
        self.max_speakers = 4  # Maximum for telephonic (usually 2-3 + possible IVR)
        self.min_speakers = 2  # Minimum for telephonic

        # Work directory
        self.work_dir = f"/tmp/msdd_{self.model_instance_name}"
        os.makedirs(self.work_dir, exist_ok=True)

        # Initialize models
        self._init_models()
        
        print("[MSDD Diarizer] ✓ Ready for telephonic diarization with IVR filtering")

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

    # -------------------- DTMF/Tone Detection --------------------

    def detect_ivr_segments_comprehensive(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        Comprehensive IVR detection using energy, spectral, and timing analysis.
        Designed to catch automated prompts while avoiding false positives on human speech.
        
        Returns:
            List of (start, end) tuples for detected IVR segments
        """
        from scipy import signal as sp_signal
        
        # Compute spectrogram
        f, t, Sxx = sp_signal.spectrogram(audio, self.sample_rate, nperseg=512, noverlap=384)
        
        # === Method 1: Energy-based detection ===
        hop_samples = int((t[1] - t[0]) * self.sample_rate) if len(t) > 1 else 512
        energy = []
        energy_times = []
        
        for i in range(0, len(audio) - hop_samples, hop_samples):
            chunk = audio[i:i+hop_samples]
            chunk_energy = np.sqrt(np.mean(chunk**2))
            energy.append(chunk_energy)
            energy_times.append(i / self.sample_rate)
        
        energy = np.array(energy)
        energy_times = np.array(energy_times)
        
        # Calculate energy variance (IVR has low variance = monotone)
        window_size = 10
        energy_variance = np.zeros(len(energy))
        for i in range(window_size, len(energy) - window_size):
            window = energy[i-window_size:i+window_size]
            energy_variance[i] = np.std(window)
        
        # === Method 2: Spectral characteristics ===
        # DTMF/Tone detection
        dtmf_freq_mask = (f >= 600) & (f <= 2000)
        dtmf_band = Sxx[dtmf_freq_mask, :]
        dtmf_energy = np.mean(dtmf_band, axis=0)
        
        # Spectral flatness (synthetic speech has lower flatness)
        from scipy.stats import gmean
        spectral_flatness = gmean(Sxx + 1e-10, axis=0) / (np.mean(Sxx + 1e-10, axis=0) + 1e-10)
        
        # High-frequency energy
        high_freq_mask = f >= 3000
        high_freq_energy = np.mean(Sxx[high_freq_mask, :], axis=0)
        
        # Spectral centroid
        spectral_centroid = np.sum(f.reshape(-1, 1) * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-10)
        
        # === Method 3: Zero-crossing rate ===
        zcr = np.zeros(len(t))
        hop_length = int(self.sample_rate * (t[1] - t[0]) if len(t) > 1 else 512)
        for i in range(len(t)):
            start_sample = int(t[i] * self.sample_rate)
            end_sample = min(start_sample + hop_length, len(audio))
            if end_sample > start_sample:
                chunk = audio[start_sample:end_sample]
                zcr[i] = np.sum(np.abs(np.diff(np.sign(chunk)))) / (2 * len(chunk))
        
        # === Normalize metrics ===
        def normalize_metric(x):
            mean_x = np.mean(x)
            std_x = np.std(x)
            if std_x < 1e-10:
                return np.zeros_like(x)
            return (x - mean_x) / std_x
        
        dtmf_energy_norm = normalize_metric(dtmf_energy)
        flatness_norm = normalize_metric(spectral_flatness)
        high_freq_norm = normalize_metric(high_freq_energy)
        centroid_norm = normalize_metric(spectral_centroid)
        zcr_norm = normalize_metric(zcr)
        
        # Interpolate energy variance
        if len(energy_variance) > 1:
            energy_var_interp = np.interp(t, energy_times[:len(energy_variance)], energy_variance)
            energy_var_norm = normalize_metric(energy_var_interp)
        else:
            energy_var_norm = np.zeros_like(t)
        
        # === STRICTER Detection Logic ===
        
        # Detect CLEAR DTMF/pure tones (high confidence)
        is_dtmf = (dtmf_energy_norm > 2.0) & (flatness_norm < -1.5) & (zcr_norm > 2.0)
        
        # Detect CLEAR synthetic speech (high confidence)
        is_synthetic = (flatness_norm < -1.5) & (energy_var_norm < -1.0)
        
        # Early portion detection - MUCH MORE CONSERVATIVE
        # Only flag if MULTIPLE strong indicators in first 15 seconds
        early_cutoff = min(15.0, len(audio) / self.sample_rate * 0.25)
        early_frames = t < early_cutoff
        
        # Count strong indicators in early portion
        strong_indicators_early = np.zeros(len(t), dtype=int)
        strong_indicators_early[early_frames] = (
            (flatness_norm[early_frames] < -1.5).astype(int) +  # Very low flatness
            (energy_var_norm[early_frames] < -1.0).astype(int) +  # Very low variance
            (zcr_norm[early_frames] > 2.0).astype(int) +  # Very high ZCR
            (high_freq_norm[early_frames] > 2.0).astype(int)  # Very high freq
        )
        
        # Only flag as IVR if 3+ strong indicators present
        is_early_ivr = strong_indicators_early >= 3
        
        # Combine detections (much stricter)
        is_non_speech = is_dtmf | is_synthetic | is_early_ivr
        
        # Require sustained detection (reduce false positives)
        kernel_size = 10  # Larger kernel = more conservative
        is_non_speech_sustained = np.copy(is_non_speech)
        for i in range(kernel_size, len(is_non_speech) - kernel_size):
            # Require majority of surrounding frames to agree
            if np.sum(is_non_speech[i-kernel_size:i+kernel_size]) < kernel_size:
                is_non_speech_sustained[i] = False
        
        # === Convert to segments ===
        non_speech_segments = []
        in_non_speech = False
        start_idx = 0
        
        for i in range(len(is_non_speech_sustained)):
            if is_non_speech_sustained[i] and not in_non_speech:
                start_idx = i
                in_non_speech = True
            elif not is_non_speech_sustained[i] and in_non_speech:
                in_non_speech = False
                start_time = t[start_idx]
                end_time = t[i]
                # Require minimum 1 second (was 0.3s - too short)
                if end_time - start_time > 1.0:
                    non_speech_segments.append((start_time, end_time))
        
        if in_non_speech and len(t) > start_idx:
            start_time = t[start_idx]
            end_time = t[-1]
            if end_time - start_time > 1.0:
                non_speech_segments.append((start_time, end_time))
        
        # === Post-validation: Check if detected segments are actually IVR ===
        validated_segments = []
        for seg_start, seg_end in non_speech_segments:
            # Extract audio from this segment
            start_sample = int(seg_start * self.sample_rate)
            end_sample = int(seg_end * self.sample_rate)
            seg_audio = audio[start_sample:end_sample]
            
            # Check if it has IVR characteristics
            seg_energy = np.sqrt(np.mean(seg_audio**2))
            seg_std = np.std(seg_audio)
            
            # IVR typically has:
            # 1. Moderate-to-high energy (not silence)
            # 2. Low variance (monotone)
            # 3. Unusual spectral characteristics
            
            # If energy is too low, it's just silence - not IVR
            if seg_energy < 0.01:
                print(f"  [IVR Filter] Rejecting {seg_start:.1f}-{seg_end:.1f}s: too quiet (energy={seg_energy:.4f})")
                continue
            
            # If it's at the very beginning AND has IVR indicators, keep it
            if seg_start < 3.0 and seg_std < 0.02:
                validated_segments.append((seg_start, seg_end))
                print(f"  [IVR Filter] Validated {seg_start:.1f}-{seg_end:.1f}s: early IVR detected")
            # If it has very strong indicators anywhere, keep it
            elif seg_std < 0.01:
                validated_segments.append((seg_start, seg_end))
                print(f"  [IVR Filter] Validated {seg_start:.1f}-{seg_end:.1f}s: strong synthetic signal")
        
        if validated_segments:
            print(f"  [IVR Detection] Found {len(validated_segments)} IVR segments:")
            for start, end in validated_segments:
                print(f"    {start:.1f}s - {end:.1f}s")
        else:
            print(f"  [IVR Detection] No IVR detected - normal conversation")
        
        return validated_segments
    
    def detect_ivr_and_dtmf(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        Enhanced detection for IVR prompts and DTMF tones.
        Uses multiple detection methods to catch automated/synthetic speech.
        """
        return self.detect_ivr_segments_comprehensive(audio)
    
    def detect_non_speech_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        Wrapper that calls the comprehensive IVR/DTMF detection
        """
        return self.detect_ivr_segments_comprehensive(audio)

    def filter_vad_segments(
        self, 
        vad_segments: List[tuple], 
        non_speech_segments: List[Tuple[float, float]],
        min_duration: float = 0.5
    ) -> List[tuple]:
        """Remove non-speech segments from VAD segments"""
        
        if not non_speech_segments:
            return vad_segments
        
        filtered_segments = []
        
        for vad_start, vad_end in vad_segments:
            current_start = vad_start
            
            for ns_start, ns_end in non_speech_segments:
                # If non-speech overlaps with this VAD segment
                if ns_start < vad_end and ns_end > vad_start:
                    # Add segment before non-speech (if significant)
                    if ns_start > current_start + min_duration:
                        filtered_segments.append((current_start, ns_start))
                    
                    # Move current start after non-speech
                    current_start = max(current_start, ns_end)
            
            # Add remaining segment after all non-speech
            if current_start < vad_end and vad_end - current_start >= min_duration:
                filtered_segments.append((current_start, vad_end))
        
        return filtered_segments

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
    # FIX #2: Shorter window for better handling of short initial utterances

    def extract_embeddings(
        self, 
        audio: np.ndarray,
        segments: List[tuple],
        window_length: float = 1.0,  # CHANGED: from 1.5 to 1.0 for shorter utterances
        shift_length: float = 0.5,   # CHANGED: from 0.75 to 0.5 for better overlap
        min_segment_for_embedding: float = 0.5  # NEW: minimum segment length to attempt embedding
    ) -> tuple:
        """
        Extract speaker embeddings with improved handling of short segments.
        
        FIX: Use shorter windows and handle segments shorter than window_length
        by padding or using the full segment.
        """
        import torch
        
        embeddings = []
        embedding_times = []
        
        for seg_start, seg_end in segments:
            seg_duration = seg_end - seg_start
            
            # FIX: Handle short segments that are shorter than window_length
            if seg_duration < window_length:
                # For segments shorter than window, use the full segment if it's long enough
                if seg_duration >= min_segment_for_embedding:
                    start_sample = int(seg_start * self.sample_rate)
                    end_sample = int(seg_end * self.sample_rate)
                    
                    chunk = audio[start_sample:end_sample]
                    
                    # Pad to minimum length if needed (center padding)
                    min_samples = int(min_segment_for_embedding * self.sample_rate)
                    if len(chunk) < min_samples:
                        pad_total = min_samples - len(chunk)
                        pad_left = pad_total // 2
                        pad_right = pad_total - pad_left
                        chunk = np.pad(chunk, (pad_left, pad_right), mode='constant')
                    
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
                    embedding_times.append((seg_start, seg_end))
                    
                    print(f"  [Embedding] Short segment {seg_start:.2f}-{seg_end:.2f}s "
                          f"({seg_duration:.2f}s) - extracted with padding")
                continue
            
            # Normal windowed extraction for longer segments
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
            
            # FIX: Handle the remaining portion at the end of the segment
            remaining_duration = seg_end - current_time
            if remaining_duration >= min_segment_for_embedding and current_time < seg_end:
                start_sample = int(current_time * self.sample_rate)
                end_sample = int(seg_end * self.sample_rate)
                
                chunk = audio[start_sample:end_sample]
                
                # Pad if needed
                min_samples = int(min_segment_for_embedding * self.sample_rate)
                if len(chunk) < min_samples:
                    pad_total = min_samples - len(chunk)
                    chunk = np.pad(chunk, (0, pad_total), mode='constant')
                
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
                embedding_times.append((current_time, seg_end))
        
        if len(embeddings) == 0:
            return np.array([]), []
        
        return np.vstack(embeddings), embedding_times

    # -------------------- Speaker Detection (Conservative for Telephonic) --------------------

    def estimate_num_speakers_telephonic(
        self, 
        embeddings: np.ndarray
    ) -> int:
        """
        Conservative speaker estimation optimized for telephonic calls.
        Typically 2-3 speakers in call center scenarios.
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        from sklearn.cluster import SpectralClustering
        from sklearn.preprocessing import normalize
        
        if len(embeddings) < 2:
            return 1
        if len(embeddings) < 4:
            return 2
        
        embeddings_norm = normalize(embeddings)
        affinity = np.dot(embeddings_norm, embeddings_norm.T)
        affinity = (affinity + 1) / 2
        
        # CRITICAL FIX: Ensure diagonal is exactly 1.0 for affinity
        np.fill_diagonal(affinity, 1.0)
        
        # For silhouette_score with metric='precomputed', we need DISTANCE not similarity
        # Convert affinity (similarity) to distance
        distance = 1.0 - affinity
        np.fill_diagonal(distance, 0.0)  # Distance from point to itself is 0
        
        scores = {}
        
        print(f"\n  [Auto-detect] Testing {self.min_speakers}-{min(self.max_speakers, len(embeddings))} speakers:")
        
        for n in range(self.min_speakers, min(self.max_speakers + 1, len(embeddings))):
            try:
                clustering = SpectralClustering(
                    n_clusters=n,
                    affinity='precomputed',
                    n_init=10,
                    random_state=42
                )
                labels = clustering.fit_predict(affinity)
                
                unique_labels = len(set(labels))
                if unique_labels < n:
                    continue
                
                # Silhouette score with DISTANCE matrix
                sil_score = silhouette_score(distance, labels, metric='precomputed')
                
                # Davies-Bouldin score (uses raw embeddings)
                db_score = davies_bouldin_score(embeddings_norm, labels)
                
                # Normalize DB score (invert so higher is better)
                normalized_db = 1.0 / (1.0 + db_score)
                
                # Combined score (favor silhouette more for telephonic)
                combined = 0.7 * sil_score + 0.3 * normalized_db
                scores[n] = combined
                
                print(f"    {n} speakers: sil={sil_score:.3f}, DB={db_score:.3f}, combined={combined:.3f}")
                
            except Exception as e:
                print(f"    {n} speakers: Failed - {e}")
                continue
        
        if not scores:
            print("  [Auto-detect] ✗ Detection failed, defaulting to 2 speakers")
            return 2
        
        best_n = max(scores.items(), key=lambda x: x[1])[0]
        
        # Conservative adjustment: for telephonic, prefer 2-3 speakers
        # If best_n > 3 and score difference is small, use 3 instead
        if best_n > 3 and len(scores) >= 3:
            score_3 = scores.get(3, 0)
            score_best = scores[best_n]
            if score_best - score_3 < 0.1:  # Within 10% confidence
                print(f"  [Auto-detect] Adjusting {best_n} → 3 (telephonic optimization)")
                best_n = 3
        
        print(f"  [Auto-detect] ✓ Selected {best_n} speakers")
        return best_n

    # -------------------- Clustering --------------------

    def cluster_embeddings(self, embeddings: np.ndarray) -> tuple:
        """Cluster embeddings - returns (labels, num_speakers)"""
        from sklearn.cluster import SpectralClustering
        from sklearn.preprocessing import normalize
        
        if len(embeddings) < 2:
            return np.zeros(len(embeddings), dtype=int), 1
        
        # Auto-detect optimal number of speakers
        num_speakers = self.estimate_num_speakers_telephonic(embeddings)
        
        # Ensure bounds
        num_speakers = max(self.min_speakers, min(num_speakers, len(embeddings)))
        
        embeddings_norm = normalize(embeddings)
        affinity = np.dot(embeddings_norm, embeddings_norm.T)
        affinity = (affinity + 1) / 2
        
        # CRITICAL FIX: Ensure diagonal is exactly 1.0
        np.fill_diagonal(affinity, 1.0)
        
        try:
            clustering = SpectralClustering(
                n_clusters=num_speakers,
                affinity='precomputed',
                n_init=10,
                random_state=42
            )
            labels = clustering.fit_predict(affinity)
        except Exception as e:
            print(f"  [Clustering] Failed, using fallback: {e}")
            labels = np.zeros(len(embeddings), dtype=int)
            chunk_size = len(embeddings) // num_speakers
            for i in range(num_speakers):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < num_speakers - 1 else len(embeddings)
                labels[start_idx:end_idx] = i
        
        # CRITICAL: Re-map labels to be sequential starting from 0
        # This fixes the speaker_2, speaker_4, speaker_5 issue
        unique_labels = sorted(set(labels))
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_mapping[label] for label in labels])
        num_speakers = len(unique_labels)
        
        print(f"  [Clustering] Remapped to {num_speakers} sequential speakers (0-{num_speakers-1})")
        
        return labels, num_speakers

    # -------------------- MSDD Refinement --------------------

    def run_msdd(
        self,
        audio: np.ndarray,
        embeddings: np.ndarray,
        embedding_times: List[tuple],
        initial_labels: np.ndarray,
        num_speakers: int
    ) -> List[Dict[str, Any]]:
        """MSDD refinement"""
        import torch
        
        if len(embeddings) == 0:
            return []
        
        # Convert audio to tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long, device=self.device)
        
        # Get audio features from MSDD encoder
        with torch.no_grad():
            try:
                # Option 1: Use preprocessor + forward methods
                processed_signal, processed_length = self.msdd_model.preprocessor(
                    input_signal=audio_tensor,
                    length=audio_length
                )
                
                # MSDD models typically use forward_encoder or _forward_encoder
                if hasattr(self.msdd_model, 'forward_encoder'):
                    encoded, encoded_len = self.msdd_model.forward_encoder(
                        audio_signal=processed_signal,
                        length=processed_length
                    )
                elif hasattr(self.msdd_model, '_forward_encoder'):
                    encoded, encoded_len = self.msdd_model._forward_encoder(
                        audio_signal=processed_signal,
                        length=processed_length
                    )
                elif hasattr(self.msdd_model, 'encoder'):
                    encoded, encoded_len = self.msdd_model.encoder(
                        audio_signal=processed_signal,
                        length=processed_length
                    )
                else:
                    # Fallback: just use preprocessor output
                    print(f"[WARNING] Could not find encoder method, using preprocessor output")
                    encoded = processed_signal
                    encoded_len = processed_length
                    
            except Exception as e:
                print(f"[WARNING] MSDD feature extraction failed: {e}")
                return self._labels_to_segments(embedding_times, initial_labels)
        
        features = encoded
        feature_length = encoded_len
        
        # Create timestamps
        total_duration = len(audio) / self.sample_rate
        feature_frames = features.shape[1]
        frame_duration = total_duration / feature_frames
        
        timestamps = np.zeros((1, feature_frames, 2), dtype=np.float32)
        for i in range(feature_frames):
            timestamps[0, i, 0] = i * frame_duration
            timestamps[0, i, 1] = (i + 1) * frame_duration
        ms_seg_timestamps = torch.from_numpy(timestamps).to(self.device)
        
        # Scale configuration
        num_scales = 5
        scale_mapping = [[0.5, 0.25], [0.75, 0.375], [1.0, 0.5], [1.25, 0.625], [1.5, 0.75]]
        seg_counts = np.full((1, num_scales), feature_frames, dtype=np.int64)
        ms_seg_counts = torch.from_numpy(seg_counts).to(self.device)
        
        # Map labels to frames
        clus_labels = torch.zeros(1, feature_frames, dtype=torch.long, device=self.device)
        for i, (start, end) in enumerate(embedding_times):
            label = initial_labels[i]
            start_frame = int(start / frame_duration)
            end_frame = int(end / frame_duration)
            end_frame = min(end_frame, feature_frames)
            clus_labels[0, start_frame:end_frame] = label
        
        # Create targets
        targets = torch.zeros(1, feature_frames, num_speakers, device=self.device)
        for i in range(feature_frames):
            label = clus_labels[0, i].item()
            if label < num_speakers:
                targets[0, i, label] = 1.0
        
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
                
                if isinstance(output, tuple):
                    probs = output[0]
                else:
                    probs = output
                
                if probs.dim() == 3:
                    probs = probs[0].cpu().numpy()
                elif probs.dim() == 2:
                    probs = probs.cpu().numpy()
                else:
                    return self._labels_to_segments(embedding_times, initial_labels)
                
            except Exception as e:
                print(f"[WARNING] MSDD forward failed: {e}")
                return self._labels_to_segments(embedding_times, initial_labels)
        
        segments = self._msdd_to_segments_from_frames(probs, frame_duration, num_speakers)
        return segments

    def _msdd_to_segments_from_frames(
        self,
        probs: np.ndarray,
        frame_duration: float,
        num_speakers: int,
        threshold: float = 0.5,
        min_duration: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Convert MSDD probabilities to segments"""
        
        if probs.ndim != 2:
            return []
        
        if probs.shape[1] != num_speakers:
            if probs.shape[1] < num_speakers:
                padding = np.zeros((probs.shape[0], num_speakers - probs.shape[1]))
                probs = np.hstack([probs, padding])
            else:
                probs = probs[:, :num_speakers]
        
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
        """Fallback: convert labels to segments"""
        
        if len(labels) == 0:
            return []
        
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
    # FIX #1: Conditional IVR removal - only when IVR is actually detected

    def remove_initial_ivr_segments_conditional(
        self,
        segments: List[Dict[str, Any]],
        ivr_was_detected: bool,
        ivr_segments: List[Tuple[float, float]],
        max_ivr_cutoff: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        CONDITIONALLY remove IVR segments at the beginning.
        
        FIX: Only removes segments when IVR was actually detected by the IVR detector.
        This prevents removing legitimate conversation segments when there's no IVR.
        
        Args:
            segments: List of speaker segments
            ivr_was_detected: Boolean indicating if IVR was detected
            ivr_segments: List of detected IVR (start, end) tuples
            max_ivr_cutoff: Maximum time to consider for IVR removal
            
        Returns:
            Filtered segments with renumbered speakers (only if IVR detected)
        """
        if not segments:
            return segments
        
        # FIX: Only apply IVR removal if IVR was actually detected
        if not ivr_was_detected or not ivr_segments:
            print(f"  [IVR Removal] Skipping - no IVR detected, keeping all segments")
            return segments
        
        # Calculate the actual IVR end time from detected segments
        ivr_end_time = max(seg[1] for seg in ivr_segments)
        actual_cutoff = min(ivr_end_time + 0.5, max_ivr_cutoff)  # Add 0.5s buffer
        
        print(f"  [IVR Removal] IVR detected ending at {ivr_end_time:.1f}s, "
              f"using cutoff {actual_cutoff:.1f}s")
        
        # Find first segment that extends beyond the IVR cutoff
        first_real_speech_idx = None
        for i, seg in enumerate(segments):
            if seg['end'] > actual_cutoff:
                first_real_speech_idx = i
                break
        
        if first_real_speech_idx is None:
            # All segments are before cutoff - keep everything as fallback
            print(f"  [IVR Removal] Warning: All segments before cutoff, keeping all")
            return segments
        
        if first_real_speech_idx == 0:
            # No segments to remove
            return segments
        
        # Remove IVR segments
        print(f"  [IVR Removal] Removing {first_real_speech_idx} initial segments (IVR)")
        for i in range(first_real_speech_idx):
            print(f"    Removed: {segments[i]['speaker']} at {segments[i]['start']:.1f}-{segments[i]['end']:.1f}s")
        
        remaining_segments = segments[first_real_speech_idx:]
        
        # Clip first remaining segment if it starts before cutoff
        if remaining_segments and remaining_segments[0]['start'] < actual_cutoff:
            print(f"  [IVR Removal] Clipping first segment start: "
                  f"{remaining_segments[0]['start']:.1f}s → {actual_cutoff:.1f}s")
            remaining_segments[0]['start'] = actual_cutoff
            remaining_segments[0]['duration'] = round(
                remaining_segments[0]['end'] - remaining_segments[0]['start'], 3
            )
        
        # Get unique speakers and renumber sequentially
        remaining_speakers = sorted(set(s['speaker'] for s in remaining_segments))
        speaker_map = {}
        for new_id, old_speaker in enumerate(remaining_speakers):
            speaker_map[old_speaker] = f"speaker_{new_id}"
        
        # Apply renumbering
        for seg in remaining_segments:
            old_speaker = seg['speaker']
            seg['speaker'] = speaker_map[old_speaker]
            if old_speaker != seg['speaker']:
                print(f"  [Renumber] {old_speaker} → {seg['speaker']}")
        
        print(f"  [IVR Removal] Retained {len(remaining_segments)} segments "
              f"with {len(remaining_speakers)} speakers")
        
        return remaining_segments

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
        """Main diarization pipeline with FIXED IVR handling"""
        import time
        
        duration = len(audio) / self.sample_rate
        
        print(f"\n{'='*80}")
        print(f"[{request_id}] Processing {duration:.2f}s audio")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            print(f"[{request_id}] Step 0: Detecting non-speech (DTMF/IVR/tones)...")
            non_speech_segments = self.detect_non_speech_segments(audio)
            ivr_was_detected = len(non_speech_segments) > 0  # FIX: Track if IVR was actually detected
            print(f"[{request_id}]   ✓ Found {len(non_speech_segments)} non-speech segments to exclude")
            for i, (start, end) in enumerate(non_speech_segments[:5]):
                print(f"[{request_id}]      Non-speech {i+1}: {start:.1f}s - {end:.1f}s")
            
            print(f"[{request_id}] Step 1: Running VAD...")
            vad_segments = self.run_vad(audio)
            print(f"[{request_id}]   ✓ Found {len(vad_segments)} raw speech segments")
            
            vad_segments = self.filter_vad_segments(vad_segments, non_speech_segments)
            print(f"[{request_id}]   ✓ After filtering: {len(vad_segments)} speech segments")
            
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
            initial_labels, num_speakers = self.cluster_embeddings(embeddings)
            print(f"[{request_id}]   ✓ Detected {num_speakers} speakers")
            
            print(f"[{request_id}] Step 4: Running MSDD refinement...")
            # TEMPORARILY DISABLED: MSDD has compatibility issues with this model version
            # Using clustering-based segmentation instead (more reliable)
            print(f"[{request_id}]   ℹ MSDD disabled - using clustering fallback")
            segments = self._labels_to_segments(embedding_times, initial_labels)
            print(f"[{request_id}]   ✓ Generated {len(segments)} segments from clustering")
            
            # FIX #1: Only remove IVR segments if IVR was actually detected
            print(f"[{request_id}] Step 5: Conditional IVR removal (only if detected)...")
            segments = self.remove_initial_ivr_segments_conditional(
                segments, 
                ivr_was_detected=ivr_was_detected,
                ivr_segments=non_speech_segments,
                max_ivr_cutoff=10.0
            )
            print(f"[{request_id}]   ✓ After IVR removal: {len(segments)} segments")
            
            print(f"[{request_id}] Step 6: Merging adjacent segments...")
            segments = self.merge_segments(segments)
            
            inference_time = time.time() - start_time
            final_speakers = set(s["speaker"] for s in segments) if segments else set()
            
            print(f"[{request_id}] ✓ Complete: {len(segments)} segments, "
                  f"{len(final_speakers)} speakers ({inference_time:.2f}s)")
            print(f"{'='*80}\n")
            
            return {
                "segments": segments,
                "num_speakers": len(final_speakers),
                "total_segments": len(segments),
                "duration": round(duration, 3),
                "inference_time": round(inference_time, 3),
                "non_speech_segments_filtered": len(non_speech_segments),
                "ivr_detected": ivr_was_detected,  # NEW: Include detection flag
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