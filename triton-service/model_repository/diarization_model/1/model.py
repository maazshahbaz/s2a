"""
Triton Inference Server - Concurrent Batch Processing
Processes entire batch concurrently without for loops
Model: NeMo diar_msdd_telephonic
"""

import os
import sys
import json
import numpy as np
import soundfile as sf
import warnings
import logging
from typing import List, Dict, Any
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

# Triton imports
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Python Backend for Speaker Diarization - Concurrent Processing
    
    Batch Configuration:
    - max_batch_size: 10
    - instances: 10
    - All requests in batch processed concurrently
    """
    
    def initialize(self, args):
        """
        Initialize the model on Triton server startup
        
        Args:
            args: Dictionary with model configuration
        """
        import torch
        from nemo.collections.asr.models import ClusteringDiarizer
        from omegaconf import OmegaConf
        
        # Suppress NeMo logging
        for logger_name in ['nemo_logger', 'nemo', 'pytorch_lightning']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
            logging.getLogger(logger_name).disabled = True
        
        # Get model config
        self.model_config = json.loads(args['model_config'])
        self.model_instance_name = args['model_instance_name']
        self.model_instance_device_id = args['model_instance_device_id']
        
        # Setup device
        self.device = f"cuda:{self.model_instance_device_id}" if torch.cuda.is_available() else "cpu"
        
        print(f"[Triton Diarization Concurrent] Initializing on {self.device}")
        
        # Enable GPU optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        # Create temporary working directory
        self.work_dir = f"/tmp/diarization_work_{self.model_instance_name}"
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Thread pool for concurrent processing
        # Each instance can process batch_size (10) requests concurrently
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Thread-local storage for diarizers (one per thread)
        self.thread_local = threading.local()
        
        # Pre-create diarization config template
        self.config_template = self._get_config_template()
        
        print(f"[Triton Diarization Concurrent] Instance {self.model_instance_name} ready on {self.device}")
    
    def _suppress_output(self):
        """Context manager to suppress output"""
        class SuppressOutput:
            def __enter__(self):
                self.old_stdout = sys.stdout
                self.old_stderr = sys.stderr
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
                return self
            
            def __exit__(self, *args):
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = self.old_stdout
                sys.stderr = self.old_stderr
        
        return SuppressOutput()
    
    def _get_config_template(self):
        """Get diarization configuration template"""
        from omegaconf import OmegaConf
        
        config_dict = {
            'device': self.device,
            'verbose': False,
            'num_workers': 0,
            'sample_rate': 16000,
            'batch_size': 64,
            
            'diarizer': {
                'manifest_filepath': None,  # Will be set per request
                'out_dir': None,  # Will be set per request
                'oracle_vad': False,
                'ignore_overlap': True,
                'collar': 0.25,
                
                'speaker_embeddings': {
                    'model_path': 'titanet_large',
                    'parameters': {
                        'window_length_in_sec': [1.5],
                        'shift_length_in_sec': [0.75],
                        'multiscale_weights': [1],
                        'save_embeddings': False,
                    }
                },
                
                'clustering': {
                    'parameters': {
                        'oracle_num_speakers': True,
                        'max_num_speakers': 2,
                        'enhanced_count_thres': 80,
                        'max_rp_threshold': 0.25,
                        'sparse_search_volume': 30,
                    }
                },
                
                'msdd_model': {
                    'model_path': 'diar_msdd_telephonic',
                    'parameters': {
                        'use_speaker_model_from_ckpt': True,
                        'infer_batch_size': 128,
                        'sigmoid_threshold': [0.7],
                        'seq_eval_mode': False,
                        'split_infer': True,
                        'diar_window_length': 50,
                        'overlap_infer_spk_limit': 5,
                    }
                },
                
                'vad': {
                    'model_path': 'vad_multilingual_marblenet',
                    'parameters': {
                        'window_length_in_sec': 0.15,
                        'shift_length_in_sec': 0.01,
                        'smoothing': 'median',
                        'overlap': 0.5,
                        'onset': 0.4,
                        'offset': 0.7,
                        'pad_onset': 0.05,
                        'pad_offset': -0.1,
                        'min_duration_on': 0.2,
                        'min_duration_off': 0.2,
                        'filter_speech_first': True,
                    }
                },
            }
        }
        
        return config_dict
    
    def _get_thread_diarizer(self):
        """Get or create diarizer for current thread"""
        if not hasattr(self.thread_local, 'diarizer'):
            from nemo.collections.asr.models import ClusteringDiarizer
            from omegaconf import OmegaConf
            
            # Create config for this thread
            config = OmegaConf.create(self.config_template)
            
            # Create diarizer
            with self._suppress_output():
                self.thread_local.diarizer = ClusteringDiarizer(cfg=config)
        
        return self.thread_local.diarizer
    
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int, request_id: str) -> str:
        """
        Preprocess audio and save to temp file
        
        Args:
            audio_data: Audio waveform
            sample_rate: Original sample rate
            request_id: Unique request identifier
            
        Returns:
            Path to preprocessed audio file
        """
        import librosa
        
        # Convert to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Save to temp file
        temp_dir = os.path.join(self.work_dir, request_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, "audio.wav")
        sf.write(temp_path, audio_data, sample_rate, subtype='PCM_16')
        
        return temp_path, temp_dir
    
    def create_manifest(self, audio_path: str, work_dir: str):
        """Create manifest file"""
        manifest_path = os.path.join(work_dir, 'manifest.json')
        
        manifest_data = {
            'audio_filepath': os.path.abspath(audio_path),
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'text': '-',
            'num_speakers': 2,
            'rttm_filepath': None,
            'uem_filepath': None
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f)
            f.write('\n')
        
        return manifest_path
    
    def parse_rttm(self, rttm_path: str) -> List[Dict[str, Any]]:
        """Parse RTTM file"""
        segments = []
        
        if not os.path.exists(rttm_path):
            return segments
        
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    speaker = parts[7]
                    start = float(parts[3])
                    duration = float(parts[4])
                    end = start + duration
                    
                    segments.append({
                        'speaker': speaker,
                        'start': start,
                        'end': end,
                        'duration': duration
                    })
        
        return segments
    
    def process_single_request(self, audio_data: np.ndarray, sample_rate: int, request_id: str) -> Dict[str, Any]:
        """
        Process a single audio file
        
        Args:
            audio_data: Audio waveform
            sample_rate: Sample rate
            request_id: Unique identifier
            
        Returns:
            Diarization results
        """
        try:
            # Preprocess audio
            audio_path, work_dir = self.preprocess_audio(audio_data, sample_rate, request_id)
            
            # Create manifest
            manifest_path = self.create_manifest(audio_path, work_dir)
            
            # Get thread-local diarizer
            diarizer = self._get_thread_diarizer()
            
            # Update config for this request
            diarizer._cfg.diarizer.manifest_filepath = manifest_path
            diarizer._cfg.diarizer.out_dir = work_dir
            
            # Run diarization (suppressed)
            with self._suppress_output():
                diarizer.diarize()
            
            # Find RTTM file
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            possible_rttm_files = [
                os.path.join(work_dir, "pred_rttms", f"{audio_basename}.rttm"),
                os.path.join(work_dir, f"{audio_basename}.rttm"),
            ]
            
            rttm_file = None
            for possible_file in possible_rttm_files:
                if os.path.exists(possible_file):
                    rttm_file = possible_file
                    break
            
            # Parse results
            if rttm_file and os.path.exists(rttm_file):
                segments = self.parse_rttm(rttm_file)
                
                result = {
                    'segments': segments,
                    'num_speakers': len(set(s['speaker'] for s in segments)),
                    'total_segments': len(segments),
                    'status': 'success'
                }
            else:
                result = {
                    'error': 'RTTM file not found',
                    'status': 'error'
                }
            
            # Cleanup temp files
            import shutil
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def execute(self, requests):
        """
        Process a batch of inference requests CONCURRENTLY
        
        Args:
            requests: List of pb_utils.InferenceRequest (batch)
            
        Returns:
            List of pb_utils.InferenceResponse
        """
        import torch
        
        batch_size = len(requests)
        
        # Extract all inputs from batch
        batch_data = []
        for idx, request in enumerate(requests):
            audio_input = pb_utils.get_input_tensor_by_name(request, "audio_input")
            sample_rate_input = pb_utils.get_input_tensor_by_name(request, "sample_rate")
            
            audio_data = audio_input.as_numpy()
            sample_rate_data = sample_rate_input.as_numpy()
            
            # Handle batched input shapes
            # audio_data might be [batch, length] or [length]
            if len(audio_data.shape) == 2:
                # Batched: [1, length] -> [length]
                audio_data = audio_data[0]
            
            # sample_rate might be [batch, 1] or [1]
            if len(sample_rate_data.shape) == 2:
                sample_rate = int(sample_rate_data[0, 0])
            elif len(sample_rate_data.shape) == 1:
                sample_rate = int(sample_rate_data[0])
            else:
                sample_rate = int(sample_rate_data)
            
            request_id = f"{self.model_instance_name}_{uuid.uuid4().hex[:8]}"
            
            batch_data.append((audio_data, sample_rate, request_id, idx))
        
        # Process all requests concurrently using ThreadPoolExecutor
        futures = {}
        for audio_data, sample_rate, request_id, idx in batch_data:
            future = self.executor.submit(
                self.process_single_request,
                audio_data,
                sample_rate,
                request_id
            )
            futures[future] = idx
        
        # Collect results in order
        results = [None] * batch_size
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                results[idx] = {
                    'error': str(e),
                    'status': 'error'
                }
        
        # Create response tensors
        responses = []
        for result in results:
            try:
                result_json = json.dumps(result)
                
                output_tensor = pb_utils.Tensor(
                    "diarization_output",
                    np.array([result_json], dtype=object)
                )
                
                if result.get('status') == 'error':
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[output_tensor],
                        error=pb_utils.TritonError(result.get('error', 'Unknown error'))
                    )
                else:
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[output_tensor]
                    )
                
                responses.append(inference_response)
                
            except Exception as e:
                error_msg = f"Error creating response: {str(e)}"
                error_tensor = pb_utils.Tensor(
                    "diarization_output",
                    np.array([json.dumps({'error': error_msg, 'status': 'error'})], dtype=object)
                )
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[error_tensor],
                    error=pb_utils.TritonError(error_msg)
                )
                responses.append(error_response)
        
        return responses
    
    def finalize(self):
        """Cleanup on server shutdown"""
        print(f"[Triton Diarization Concurrent] Shutting down instance {self.model_instance_name}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Cleanup temp directory
        import shutil
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)