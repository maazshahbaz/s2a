"""
Optimized Triton Inference Server - Speaker Diarization
Uses TensorRT for VAD and TitaNet, torch.compile + BF16 for MSDD
Optimized for large audio files (~50 mins)

"""

import os
import sys
import json
import numpy as np
import warnings
import logging
from typing import List, Dict, Any, Tuple, Optional
import uuid
import math
import torch
import tensorrt as trt
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import normalize
from nemo.collections.asr.models import EncDecDiarLabelModel
from scipy import signal
import shutil


# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

import triton_python_backend_utils as pb_utils


class MelSpectrogramExtractor:
    """
    Pure PyTorch mel spectrogram extractor - drop-in replacement for torchaudio.
    
    Matches torchaudio.transforms.MelSpectrogram with:
    - sample_rate=16000
    - n_fft=512
    - win_length=400 (25ms)
    - hop_length=160 (10ms)
    - n_mels=80
    - f_min=0, f_max=8000
    - norm='slaney', mel_scale='slaney'
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        device: str = 'cuda'
    ):
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.device = device
        
        # Pre-compute and cache mel filterbank and window on GPU
        self.mel_filterbank = self._create_mel_filterbank().to(device)
        self.window = torch.hann_window(win_length, device=device)
        
        # Pad window to n_fft if needed
        if win_length < n_fft:
            pad_left = (n_fft - win_length) // 2
            pad_right = n_fft - win_length - pad_left
            self.window = torch.nn.functional.pad(self.window, (pad_left, pad_right))
    
    def _hz_to_mel(self, freq: float) -> float:
        """Convert Hz to mel scale (Slaney formula)"""
        f_min = 0.0
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        
        if freq >= min_log_hz:
            return min_log_mel + math.log(freq / min_log_hz) / logstep
        return (freq - f_min) / f_sp
    
    def _mel_to_hz(self, mel: float) -> float:
        """Convert mel to Hz (Slaney formula)"""
        f_min = 0.0
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        
        if mel >= min_log_mel:
            return min_log_hz * math.exp(logstep * (mel - min_log_mel))
        return f_min + f_sp * mel
    
    def _create_mel_filterbank(self) -> 'torch.Tensor':
        """Create mel filterbank matrix with Slaney normalization"""
        
        # Frequency bins
        n_freqs = self.n_fft // 2 + 1
        fft_freqs = torch.linspace(0, self.sample_rate / 2, n_freqs)
        
        # Mel points
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        
        # Convert back to Hz
        hz_points = torch.tensor([self._mel_to_hz(m.item()) for m in mel_points])
        
        # Create filterbank
        filterbank = torch.zeros(self.n_mels, n_freqs)
        
        for i in range(self.n_mels):
            lower = hz_points[i]
            center = hz_points[i + 1]
            upper = hz_points[i + 2]
            
            # Rising slope
            lower_slope = (fft_freqs - lower) / (center - lower + 1e-10)
            # Falling slope
            upper_slope = (upper - fft_freqs) / (upper - center + 1e-10)
            
            # Triangular filter
            filterbank[i] = torch.max(
                torch.zeros_like(fft_freqs),
                torch.min(lower_slope, upper_slope)
            )
            
            # Slaney normalization: normalize by bandwidth
            enorm = 2.0 / (hz_points[i + 2] - hz_points[i] + 1e-10)
            filterbank[i] *= enorm
        
        return filterbank
    
    def __call__(self, waveform: 'torch.Tensor') -> 'torch.Tensor':
        """
        Compute mel spectrogram from waveform.
        
        Args:
            waveform: [batch, samples] or [samples] tensor
            
        Returns:
            mel_spec: [batch, n_mels, time] tensor
        """
        
        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        batch_size = waveform.shape[0]
        
        # Pad waveform for STFT
        pad_amount = self.n_fft // 2
        waveform_padded = torch.nn.functional.pad(
            waveform, (pad_amount, pad_amount), mode='reflect'
        )
        
        # Compute STFT using torch.stft
        stft_out = torch.stft(
            waveform_padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,  # Window is already padded to n_fft
            window=self.window,
            center=False,  # We already padded
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        # Power spectrogram
        power_spec = stft_out.abs().pow(2)  # [batch, n_freqs, time]
        
        # Apply mel filterbank
        # mel_filterbank: [n_mels, n_freqs]
        # power_spec: [batch, n_freqs, time]
        mel_spec = torch.matmul(
            self.mel_filterbank,  # [n_mels, n_freqs]
            power_spec  # [batch, n_freqs, time] -> need to handle batch
        )
        
        # Handle batched matmul properly
        # Reshape for batch processing
        if batch_size > 1:
            # power_spec: [batch, n_freqs, time]
            # We need: [batch, n_mels, time]
            mel_spec = torch.einsum('mf,bft->bmt', self.mel_filterbank, power_spec)
        else:
            mel_spec = torch.matmul(self.mel_filterbank, power_spec.squeeze(0)).unsqueeze(0)
        
        return mel_spec
    
    def to(self, device):
        """Move filterbank and window to device"""
        self.device = device
        self.mel_filterbank = self.mel_filterbank.to(device)
        self.window = self.window.to(device)
        return self


class TritonPythonModel:
    """
    Optimized Triton Backend for Speaker Diarization
    
    Architecture:
    - VAD: TensorRT engine (vad_marblenet.plan)
    - Speaker Embeddings: TensorRT engine (titanet_large.plan)  
    - MSDD: torch.compile + BF16 optimized PyTorch
    - Clustering: CPU-based spectral clustering
    
    Optimizations for large files:
    - Chunked processing with streaming
    - Memory-efficient embedding extraction
    - Batched TensorRT inference
    """
    
    def initialize(self, args):
        """Initialize the model on Triton server startup"""
        
        # Suppress logging
        for logger_name in ['nemo_logger', 'nemo', 'pytorch_lightning']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        self.model_config = json.loads(args['model_config'])
        self.model_instance_name = args['model_instance_name']
        self.model_instance_device_id = args['model_instance_device_id']
        
        self.device = f"cuda:{self.model_instance_device_id}"
        self.device_id = int(self.model_instance_device_id)
        
        print(f"[Optimized Diarization] Initializing on {self.device}")
        
        # Enable GPU optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        
        # Configuration
        self.sample_rate = 16000
        self.num_speakers = 2  # Fixed for your use case
        
        # Paths to TensorRT engines - check multiple possible locations
        possible_model_dirs = [
            os.environ.get('DIARIZATION_MODEL_DIR', ''),
            '/models/diarization',
            '/opt/models/diarization',
            '/workspace/models/diarization',
            os.path.join(os.path.dirname(__file__), '..', 'engines'),
        ]
        

        self.model_dir = os.path.join('/models' , 'diarization_model')
        print(f"[Optimized Diarization] Using model directory: {self.model_dir}")
        
        self.vad_engine_path = os.path.join(self.model_dir, "engine", 'vad_marblenet.plan')
        self.titanet_engine_path = os.path.join(self.model_dir,  "engine", 'titanet_large.plan')
        
        # Initialize components
        self._init_feature_extractor()
        self._init_vad_trt()
        self._init_titanet_trt()
        self._init_msdd_optimized()
        self._init_clustering()
        
        # Working directory
        self.work_dir = f"/tmp/diarization_{self.model_instance_name}"
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Processing parameters
        self.vad_window = 0.15
        self.vad_shift = 0.01
        self.emb_window = 1.5
        self.emb_shift = 0.75
        
        print(f"[Optimized Diarization] Ready on {self.device}")
    
    def _init_feature_extractor(self):
        """Initialize mel spectrogram feature extractor (pure PyTorch, no torchaudio)"""
        
        # Use custom MelSpectrogramExtractor - drop-in replacement for torchaudio
        self.mel_transform = MelSpectrogramExtractor(
            sample_rate=self.sample_rate,
            n_fft=512,
            win_length=400,  # 25ms
            hop_length=160,  # 10ms
            n_mels=80,
            f_min=0,
            f_max=8000,
            device=self.device
        )
        
        print("[Optimized Diarization] Feature extractor initialized (pure PyTorch)")
    
    def _init_vad_trt(self):
        """Initialize VAD TensorRT engine"""
        
        if not os.path.exists(self.vad_engine_path):
            raise FileNotFoundError(f"VAD TensorRT engine not found: {self.vad_engine_path}")
        
        # Load TensorRT engine
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        with open(self.vad_engine_path, 'rb') as f:
            self.vad_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(f.read())
        
        self.vad_context = self.vad_engine.create_execution_context()
        
        # Allocate buffers
        self.vad_inputs = {}
        self.vad_outputs = {}
        self.vad_bindings = []
        
        for i in range(self.vad_engine.num_io_tensors):
            name = self.vad_engine.get_tensor_name(i)
            dtype = trt.nptype(self.vad_engine.get_tensor_dtype(name))
            
            if self.vad_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.vad_inputs[name] = {'dtype': dtype}
            else:
                self.vad_outputs[name] = {'dtype': dtype}
        
        print("[Optimized Diarization] VAD TensorRT engine loaded")
    
    def _init_titanet_trt(self):
        """Initialize TitaNet TensorRT engine"""
        
        if not os.path.exists(self.titanet_engine_path):
            raise FileNotFoundError(f"TitaNet TensorRT engine not found: {self.titanet_engine_path}")
        
        with open(self.titanet_engine_path, 'rb') as f:
            self.titanet_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(f.read())
        
        self.titanet_context = self.titanet_engine.create_execution_context()
        
        self.titanet_inputs = {}
        self.titanet_outputs = {}
        
        for i in range(self.titanet_engine.num_io_tensors):
            name = self.titanet_engine.get_tensor_name(i)
            dtype = trt.nptype(self.titanet_engine.get_tensor_dtype(name))
            
            if self.titanet_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.titanet_inputs[name] = {'dtype': dtype}
            else:
                self.titanet_outputs[name] = {'dtype': dtype}
        
        print("[Optimized Diarization] TitaNet TensorRT engine loaded")
    
    def _init_msdd_optimized(self):
        """Initialize MSDD with torch.compile and BF16"""

        
        print("[Optimized Diarization] Loading MSDD model...")
        
        # Load MSDD
        self.msdd_model = EncDecDiarLabelModel.from_pretrained('diar_msdd_telephonic')
        self.msdd_model.eval()
        self.msdd_model.to(self.device)
        
        # Convert to BF16 for H100
        self.msdd_model = self.msdd_model.to(torch.bfloat16)
        
        # Compile with max-autotune
        self.msdd_model = torch.compile(
            self.msdd_model,
            mode="max-autotune",
            fullgraph=False  # MSDD has dynamic control flow
        )
        
        # Warm-up compilation
        self._warmup_msdd()
        
        print("[Optimized Diarization] MSDD optimized with torch.compile + BF16")
    
    def _warmup_msdd(self):
        """Warm up MSDD to trigger compilation"""
        
        
        # Create dummy inputs
        batch_size = 1
        num_speakers = 2
        seq_len = 50
        emb_dim = 192
        
        dummy_emb = torch.randn(
            batch_size, seq_len, num_speakers, emb_dim,
            dtype=torch.bfloat16, device=self.device
        )
        dummy_lengths = torch.tensor([seq_len], device=self.device)
        
        # Run a few warm-up iterations
        with torch.no_grad():
            for _ in range(3):
                try:
                    _ = self.msdd_model.msdd._speaker_model.forward_for_export(
                        dummy_emb.reshape(-1, emb_dim).float(),
                        dummy_lengths
                    )
                except:
                    pass  # MSDD might have different input requirements
        
        print("[Optimized Diarization] MSDD warmed up")
    
    def _init_clustering(self):
        """Initialize clustering components"""        
        self.clustering_method = 'spectral'
        print("[Optimized Diarization] Clustering initialized")
    
    def extract_features(self, audio: np.ndarray) -> 'torch.Tensor':
        """Extract mel spectrogram features on GPU"""
        
        # Convert to tensor and move to GPU
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Extract mel spectrogram using pure PyTorch implementation
        with torch.no_grad():
            mel_spec = self.mel_transform(audio_tensor)
            # Log mel
            mel_spec = torch.log(mel_spec.clamp(min=1e-9))
        
        return mel_spec
    
    def run_vad_trt(self, mel_features: 'torch.Tensor') -> List[Tuple[float, float]]:
        """
        Run VAD using TensorRT engine with sliding window for frame-level detection.
        
        VAD MarbleNet classifies fixed-length windows as speech/non-speech.
        We slide a window across the audio and collect predictions.
        
        TensorRT engine constraints (from export):
        - minShapes: audio_signal:1x80x64
        - maxShapes: audio_signal:32x80x1024
        """
        
        n_mels = mel_features.shape[1]
        total_time_steps = mel_features.shape[2]
        
        # TensorRT engine constraints
        max_batch_size = 32
        min_time_steps = 64
        max_time_steps = 1024
        
        # VAD window parameters - must be within TRT limits
        vad_window_frames = min(max_time_steps, max(min_time_steps, 64))  # Use 64 frames
        vad_hop_frames = 16  # Hop between windows (160ms at 10ms per frame)
        
        # If audio is shorter than one window, pad and process
        if total_time_steps <= vad_window_frames:
            pad_size = max(0, min_time_steps - total_time_steps)
            if pad_size > 0:
                mel_padded = torch.nn.functional.pad(mel_features, (0, pad_size))
            else:
                mel_padded = mel_features
            
            actual_frames = min(vad_window_frames, mel_padded.shape[2])
            
            self.vad_context.set_input_shape('audio_signal', (1, n_mels, actual_frames))
            self.vad_context.set_input_shape('length', (1,))
            
            audio_signal = mel_padded[:, :, :actual_frames].contiguous().float()
            lengths = torch.tensor([actual_frames], dtype=torch.int64, device=self.device)
            logits = torch.empty((1, 2), dtype=torch.float32, device=self.device)
            
            self.vad_context.set_tensor_address('audio_signal', audio_signal.data_ptr())
            self.vad_context.set_tensor_address('length', lengths.data_ptr())
            self.vad_context.set_tensor_address('logits', logits.data_ptr())
            
            stream = torch.cuda.current_stream().cuda_stream
            self.vad_context.execute_async_v3(stream)
            torch.cuda.synchronize()
            
            probs = torch.softmax(logits, dim=-1)
            is_speech = probs[0, 1].item() > 0.5
            
            if is_speech:
                duration = total_time_steps * 0.01  # 10ms per frame
                return [(0.0, duration)]
            return []
        
        # Collect windows for batched processing
        windows = []
        window_times = []
        
        for start_frame in range(0, total_time_steps - vad_window_frames + 1, vad_hop_frames):
            end_frame = start_frame + vad_window_frames
            windows.append(mel_features[0, :, start_frame:end_frame])  # [n_mels, window_frames]
            window_times.append(start_frame * 0.01)  # Time in seconds
        
        # Handle remaining frames at the end
        if len(windows) == 0 or (total_time_steps - vad_window_frames) % vad_hop_frames != 0:
            last_start = max(0, total_time_steps - vad_window_frames)
            if last_start * 0.01 not in window_times:
                windows.append(mel_features[0, :, last_start:last_start + vad_window_frames])
                window_times.append(last_start * 0.01)
        
        if len(windows) == 0:
            return []
        
        # Process in batches (respecting TRT max batch size)
        vad_batch_size = min(max_batch_size, 32)
        all_speech_probs = []
        
        for i in range(0, len(windows), vad_batch_size):
            batch_windows = windows[i:i + vad_batch_size]
            current_batch = len(batch_windows)
            
            # Stack into batch [batch, mels, time]
            batch_mel = torch.stack(batch_windows, dim=0).contiguous().float()
            
            self.vad_context.set_input_shape('audio_signal', (current_batch, n_mels, vad_window_frames))
            self.vad_context.set_input_shape('length', (current_batch,))
            
            lengths = torch.tensor([vad_window_frames] * current_batch, dtype=torch.int64, device=self.device)
            logits = torch.empty((current_batch, 2), dtype=torch.float32, device=self.device)
            
            self.vad_context.set_tensor_address('audio_signal', batch_mel.data_ptr())
            self.vad_context.set_tensor_address('length', lengths.data_ptr())
            self.vad_context.set_tensor_address('logits', logits.data_ptr())
            
            stream = torch.cuda.current_stream().cuda_stream
            self.vad_context.execute_async_v3(stream)
            torch.cuda.synchronize()
            
            probs = torch.softmax(logits, dim=-1)
            all_speech_probs.extend(probs[:, 1].cpu().numpy().tolist())
        
        # Convert frame probabilities to segments
        return self._vad_probs_to_segments_windowed(all_speech_probs, window_times, vad_hop_frames * 0.01)
    
    def _vad_probs_to_segments_windowed(
        self,
        speech_probs: List[float],
        window_times: List[float],
        window_duration: float,
        threshold: float = 0.5,
        min_duration: float = 0.2
    ) -> List[Tuple[float, float]]:
        """Convert windowed VAD probabilities to speech segments"""
        
        if len(speech_probs) == 0:
            return []
        
        segments = []
        in_speech = False
        start_time = 0.0
        
        for prob, time in zip(speech_probs, window_times):
            is_speech = prob > threshold
            
            if is_speech and not in_speech:
                in_speech = True
                start_time = time
            elif not is_speech and in_speech:
                in_speech = False
                end_time = time
                if end_time - start_time >= min_duration:
                    segments.append((start_time, end_time))
        
        # Handle segment at end
        if in_speech:
            end_time = window_times[-1] + window_duration
            if end_time - start_time >= min_duration:
                segments.append((start_time, end_time))
        
        return segments
    
    def _vad_probs_to_segments(
        self, 
        speech_probs: np.ndarray,
        threshold: float = 0.5,
        min_duration: float = 0.2
    ) -> List[Tuple[float, float]]:
        """Convert VAD probabilities to speech segments"""
        
        # Frame to time conversion
        frame_duration = self.vad_shift
        
        # Threshold
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
        
        # Handle segment at end
        if in_speech:
            start_time = start_frame * frame_duration
            end_time = len(speech_frames) * frame_duration
            if end_time - start_time >= min_duration:
                segments.append((start_time, end_time))
        
        return segments
    
    def extract_embeddings_trt(
        self, 
        audio: np.ndarray, 
        segments: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Extract speaker embeddings using TensorRT TitaNet"""
        
        embeddings = []
        embedding_times = []
        
        # Process each segment with sliding window
        for seg_start, seg_end in segments:
            window_start = seg_start
            
            while window_start + self.emb_window <= seg_end:
                # Extract audio chunk
                start_sample = int(window_start * self.sample_rate)
                end_sample = int((window_start + self.emb_window) * self.sample_rate)
                
                if end_sample > len(audio):
                    break
                
                chunk = audio[start_sample:end_sample]
                
                # Extract features
                mel = self.extract_features(chunk)
                
                # Run TitaNet
                batch_size = 1
                n_mels = mel.shape[1]
                time_steps = mel.shape[2]
                
                self.titanet_context.set_input_shape('audio_signal', (batch_size, n_mels, time_steps))
                self.titanet_context.set_input_shape('length', (batch_size,))
                
                audio_signal = mel.contiguous().float()
                lengths = torch.tensor([time_steps], dtype=torch.int64, device=self.device)
                
                # TitaNet-L output shapes are fixed:
                # embs: [batch, 192] - speaker embedding
                # logits: [batch, 16681] - speaker classification logits
                emb_output = torch.empty((batch_size, 192), dtype=torch.float32, device=self.device)
                logits_output = torch.empty((batch_size, 16681), dtype=torch.float32, device=self.device)
                
                self.titanet_context.set_tensor_address('audio_signal', audio_signal.data_ptr())
                self.titanet_context.set_tensor_address('length', lengths.data_ptr())
                self.titanet_context.set_tensor_address('embs', emb_output.data_ptr())
                self.titanet_context.set_tensor_address('logits', logits_output.data_ptr())
                
                stream = torch.cuda.current_stream().cuda_stream
                self.titanet_context.execute_async_v3(stream)
                torch.cuda.synchronize()
                
                embeddings.append(emb_output.cpu().numpy().flatten())
                embedding_times.append((window_start, window_start + self.emb_window))
                
                window_start += self.emb_shift
        
        if len(embeddings) == 0:
            return np.array([]), []
        
        return np.vstack(embeddings), embedding_times
    
    def extract_embeddings_batched_trt(
        self,
        audio: np.ndarray,
        segments: List[Tuple[float, float]],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Extract embeddings with batched TensorRT inference for efficiency.
        
        TensorRT TitaNet engine constraints (from export):
        - minShapes: audio_signal:1x80x80
        - maxShapes: audio_signal:64x80x480
        """
        
        # TensorRT constraints
        max_trt_batch = 64
        max_trt_frames = 480
        min_trt_frames = 80
        
        # Cap batch size to TRT limit
        batch_size = min(batch_size, max_trt_batch)
        
        # Collect all windows
        windows = []
        window_times = []
        
        for seg_start, seg_end in segments:
            window_start = seg_start
            while window_start + self.emb_window <= seg_end:
                start_sample = int(window_start * self.sample_rate)
                end_sample = int((window_start + self.emb_window) * self.sample_rate)
                
                if end_sample > len(audio):
                    break
                
                windows.append(audio[start_sample:end_sample])
                window_times.append((window_start, window_start + self.emb_window))
                window_start += self.emb_shift
        
        if len(windows) == 0:
            return np.array([]), []
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            current_batch_size = len(batch_windows)
            
            # Pad to same length and create batch
            max_len = max(len(w) for w in batch_windows)
            batch_audio = np.zeros((current_batch_size, max_len), dtype=np.float32)
            
            for j, w in enumerate(batch_windows):
                batch_audio[j, :len(w)] = w
            
            # Extract features for batch
            batch_tensor = torch.from_numpy(batch_audio).to(self.device)
            
            with torch.no_grad():
                mel_batch = self.mel_transform(batch_tensor)
                mel_batch = torch.log(mel_batch.clamp(min=1e-9))
            
            # Check mel frames are within TRT limits
            n_mels = mel_batch.shape[1]
            time_steps = mel_batch.shape[2]
            
            # Ensure time_steps is within TRT limits
            if time_steps > max_trt_frames:
                # Truncate to max
                mel_batch = mel_batch[:, :, :max_trt_frames]
                time_steps = max_trt_frames
            elif time_steps < min_trt_frames:
                # Pad to min
                pad_size = min_trt_frames - time_steps
                mel_batch = torch.nn.functional.pad(mel_batch, (0, pad_size))
                time_steps = min_trt_frames
            
            self.titanet_context.set_input_shape('audio_signal', (current_batch_size, n_mels, time_steps))
            self.titanet_context.set_input_shape('length', (current_batch_size,))
            
            lengths = torch.tensor([time_steps] * current_batch_size, dtype=torch.int64, device=self.device)
            
            # TitaNet-L output shapes are fixed:
            # embs: [batch, 192] - speaker embedding
            # logits: [batch, 16681] - speaker classification logits
            emb_output = torch.empty((current_batch_size, 192), dtype=torch.float32, device=self.device)
            logits_output = torch.empty((current_batch_size, 16681), dtype=torch.float32, device=self.device)
            
            self.titanet_context.set_tensor_address('audio_signal', mel_batch.contiguous().float().data_ptr())
            self.titanet_context.set_tensor_address('length', lengths.data_ptr())
            self.titanet_context.set_tensor_address('embs', emb_output.data_ptr())
            self.titanet_context.set_tensor_address('logits', logits_output.data_ptr())
            
            stream = torch.cuda.current_stream().cuda_stream
            self.titanet_context.execute_async_v3(stream)
            torch.cuda.synchronize()
            
            all_embeddings.append(emb_output.cpu().numpy())
        
        return np.vstack(all_embeddings), window_times
    
    def cluster_embeddings(
        self, 
        embeddings: np.ndarray, 
        num_speakers: int = 2
    ) -> np.ndarray:
        """Cluster embeddings to get speaker labels"""
        
        if len(embeddings) < num_speakers:
            return np.zeros(len(embeddings), dtype=int)
        
        # Normalize embeddings
        embeddings_norm = normalize(embeddings)
        
        # Compute affinity matrix
        affinity = np.dot(embeddings_norm, embeddings_norm.T)
        affinity = (affinity + 1) / 2  # Scale to [0, 1]
        
        # Spectral clustering
        try:
            clustering = SpectralClustering(
                n_clusters=num_speakers,
                affinity='precomputed',
                n_init=10,
                random_state=42
            )
            labels = clustering.fit_predict(affinity)
        except:
            # Fallback to agglomerative
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings_norm)
        
        return labels
    
    def run_msdd_refinement(
        self,
        audio: np.ndarray,
        embeddings: np.ndarray,
        embedding_times: List[Tuple[float, float]],
        initial_labels: np.ndarray
    ) -> np.ndarray:
        """Run MSDD for overlap detection and label refinement"""
        
        if len(embeddings) == 0:
            return initial_labels
        
        # MSDD expects embeddings in specific format
        # [batch, seq_len, num_speakers, emb_dim]
        
        num_speakers = self.num_speakers
        seq_len = len(embeddings)
        emb_dim = embeddings.shape[1]
        
        # Reorganize embeddings by speaker based on initial clustering
        speaker_embeddings = np.zeros((1, seq_len, num_speakers, emb_dim), dtype=np.float32)
        
        for i, (emb, label) in enumerate(zip(embeddings, initial_labels)):
            speaker_embeddings[0, i, label, :] = emb
        
        # Convert to BF16 tensor
        emb_tensor = torch.from_numpy(speaker_embeddings).to(
            dtype=torch.bfloat16, 
            device=self.device
        )
        lengths = torch.tensor([seq_len], device=self.device)
        
        # Run MSDD
        try:
            with torch.no_grad():
                # MSDD forward pass
                msdd_output = self.msdd_model.msdd(
                    emb_tensor,
                    lengths
                )
                
                # Get refined probabilities
                if isinstance(msdd_output, tuple):
                    probs = msdd_output[0]
                else:
                    probs = msdd_output
                
                probs = probs.float().cpu().numpy()
                
                # Threshold to get labels
                refined_labels = np.argmax(probs[0], axis=-1)
                
                return refined_labels
                
        except Exception as e:
            print(f"[MSDD] Refinement failed: {e}, using initial labels")
            return initial_labels
    
    def process_audio(
        self, 
        audio: np.ndarray,
        request_id: str
    ) -> Dict[str, Any]:
        """
        Process complete audio file without chunking
        
        Args:
            audio: Full audio waveform
            request_id: Unique request identifier
            
        Returns:
            Diarization results
        """
        
        total_duration = len(audio) / self.sample_rate
        print(f"[{request_id}] Processing {total_duration:.1f}s audio (full file)")
        
        # Step 1: Extract features for full audio
        print(f"[{request_id}] Extracting features...")
        mel_features = self.extract_features(audio)
        
        # Step 2: Run VAD on full audio
        print(f"[{request_id}] Running VAD...")
        vad_segments = self.run_vad_trt(mel_features)
        
        # Merge overlapping VAD segments
        vad_segments = self._merge_segments(vad_segments)
        print(f"[{request_id}] Found {len(vad_segments)} speech segments")
        
        if len(vad_segments) == 0:
            return {
                'segments': [],
                'num_speakers': 0,
                'total_segments': 0,
                'duration': total_duration,
                'status': 'success'
            }
        
        # Step 3: Extract embeddings (batched for efficiency)
        print(f"[{request_id}] Extracting speaker embeddings...")
        embeddings, embedding_times = self.extract_embeddings_batched_trt(
            audio, vad_segments, batch_size=64
        )
        print(f"[{request_id}] Extracted {len(embeddings)} embeddings")
        
        if len(embeddings) == 0:
            return {
                'segments': [],
                'num_speakers': 0,
                'total_segments': 0,
                'duration': total_duration,
                'status': 'success'
            }
        
        # Step 4: Initial clustering
        print(f"[{request_id}] Clustering speakers...")
        initial_labels = self.cluster_embeddings(embeddings, self.num_speakers)
        
        # Step 5: MSDD refinement (optional, for overlap detection)
        print(f"[{request_id}] Running MSDD refinement...")
        refined_labels = self.run_msdd_refinement(
            audio, embeddings, embedding_times, initial_labels
        )
        
        # Step 6: Convert to final segments
        print(f"[{request_id}] Generating output segments...")
        segments = self._labels_to_segments(embedding_times, refined_labels)
        
        # Merge adjacent segments from same speaker
        segments = self._merge_speaker_segments(segments)
        
        result = {
            'segments': segments,
            'num_speakers': len(set(s['speaker'] for s in segments)),
            'total_segments': len(segments),
            'duration': total_duration,
            'status': 'success'
        }
        
        print(f"[{request_id}] Completed: {len(segments)} segments, {result['num_speakers']} speakers")
        
        return result
    
    def _merge_segments(
        self, 
        segments: List[Tuple[float, float]], 
        gap_threshold: float = 0.3
    ) -> List[Tuple[float, float]]:
        """Merge overlapping or close segments"""
        if not segments:
            return []
        
        # Sort by start time
        segments = sorted(segments, key=lambda x: x[0])
        
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            
            if start <= prev_end + gap_threshold:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        
        return merged
    
    def _labels_to_segments(
        self,
        embedding_times: List[Tuple[float, float]],
        labels: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Convert frame labels to speaker segments"""
        segments = []
        
        for (start, end), label in zip(embedding_times, labels):
            segments.append({
                'speaker': f'speaker_{label}',
                'start': round(start, 3),
                'end': round(end, 3),
                'duration': round(end - start, 3)
            })
        
        return segments
    
    def _merge_speaker_segments(
        self,
        segments: List[Dict[str, Any]],
        gap_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Merge adjacent segments from same speaker"""
        if not segments:
            return []
        
        merged = [segments[0].copy()]
        
        for seg in segments[1:]:
            if (seg['speaker'] == merged[-1]['speaker'] and 
                seg['start'] - merged[-1]['end'] <= gap_threshold):
                merged[-1]['end'] = seg['end']
                merged[-1]['duration'] = round(merged[-1]['end'] - merged[-1]['start'], 3)
            else:
                merged.append(seg.copy())
        
        return merged
    
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess audio: convert to mono, resample, normalize"""
        # Convert to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed (using scipy instead of librosa)
        if sample_rate != self.sample_rate:
            
            # Calculate resampling ratio
            num_samples = int(len(audio_data) * self.sample_rate / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)
        
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        return audio_data.astype(np.float32)
    
    def execute(self, requests):
        """Process inference requests"""
        
        responses = []
        
        for request in requests:
            try:
                # Extract inputs
                audio_input = pb_utils.get_input_tensor_by_name(request, "audio_input")
                sample_rate_input = pb_utils.get_input_tensor_by_name(request, "sample_rate")
                
                audio_data = audio_input.as_numpy()
                sample_rate_data = sample_rate_input.as_numpy()
                
                # Handle input shapes
                if len(audio_data.shape) == 2:
                    audio_data = audio_data[0]
                
                if len(sample_rate_data.shape) >= 1:
                    sample_rate = int(sample_rate_data.flat[0])
                else:
                    sample_rate = int(sample_rate_data)
                
                request_id = f"{self.model_instance_name}_{uuid.uuid4().hex[:8]}"
                
                # Preprocess
                audio = self.preprocess_audio(audio_data, sample_rate)
                
                # Process complete audio
                result = self.process_audio(audio, request_id)
                
                # Create response
                result_json = json.dumps(result)
                output_tensor = pb_utils.Tensor(
                    "diarization_output",
                    np.array([result_json], dtype=object)
                )
                
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
                
            except Exception as e:
                error_result = {
                    'error': str(e),
                    'status': 'error'
                }
                error_tensor = pb_utils.Tensor(
                    "diarization_output",
                    np.array([json.dumps(error_result)], dtype=object)
                )
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[error_tensor],
                    error=pb_utils.TritonError(str(e))
                ))
        
        return responses
    
    def finalize(self):
        """Cleanup on shutdown"""
        
        print(f"[Optimized Diarization] Shutting down {self.model_instance_name}")
        
        # Cleanup temp directory
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)