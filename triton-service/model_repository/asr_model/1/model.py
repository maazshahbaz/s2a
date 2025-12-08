import triton_python_backend_utils as pb_utils
import numpy as np
import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf
import tempfile
import os
import json
import gc
import torch
from omegaconf import OmegaConf, open_dict

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model with optimizations"""
        self.model_config = json.loads(args['model_config'])
        
        # Load the ASR model
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )
        
        # Move to GPU and set to eval mode
        self.asr_model = self.asr_model.cuda()
        self.asr_model.eval()
        
        # Check if bfloat16 is supported
        self.use_amp = torch.cuda.is_bf16_supported()
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float16
        
        self.target_sr = 16000
        
        # Enable optimizations - but DISABLE CUDA graphs due to version incompatibility
        self._enable_optimizations()
        
        # Warm up the model
        self._warmup()
        
    def _enable_optimizations(self):
        """Enable decoding optimizations - disable CUDA graphs for TDT compatibility"""
        try:
            decoding_cfg = self.asr_model.cfg.decoding
            
            with open_dict(decoding_cfg):
                # Enable label looping (provides speedup)
                # But DISABLE CUDA graphs (causes version compatibility issues with TDT)
                if hasattr(decoding_cfg, 'greedy'):
                    with open_dict(decoding_cfg.greedy):
                        decoding_cfg.greedy.loop_labels = True
                        # CRITICAL: Disable CUDA graph decoder to avoid the unpacking error
                        decoding_cfg.greedy.use_cuda_graph_decoder = False
                    print("Enabled loop_labels, DISABLED CUDA graph decoder (TDT compatibility)")
                
                decoding_cfg.strategy = "greedy_batch"
            
            self.asr_model.change_decoding_strategy(decoding_cfg)
            print("Successfully applied decoding optimizations")
            
        except Exception as e:
            print(f"Optimization method 1 failed: {e}")
            self._try_alternative_optimization()
    
    def _try_alternative_optimization(self):
        """Try alternative methods to enable optimizations"""
        try:
            if hasattr(self.asr_model, 'decoding') and self.asr_model.decoding is not None:
                decoder = self.asr_model.decoding
                
                # Enable loop_labels but disable CUDA graphs
                if hasattr(decoder, 'loop_labels'):
                    decoder.loop_labels = True
                    print("Enabled loop_labels directly on decoder")
                
                # Explicitly disable CUDA graph decoder
                if hasattr(decoder, 'use_cuda_graph_decoder'):
                    decoder.use_cuda_graph_decoder = False
                    print("Disabled CUDA graph decoder directly on decoder")
                    
                # Also check inner decoding object
                if hasattr(decoder, 'decoding'):
                    inner = decoder.decoding
                    if hasattr(inner, 'use_cuda_graph_decoder'):
                        inner.use_cuda_graph_decoder = False
                    if hasattr(inner, 'loop_labels'):
                        inner.loop_labels = True
            
        except Exception as e:
            print(f"Alternative optimization failed: {e}")
            print("Using default decoding strategy")
        
    def _warmup(self):
        """Warm up the model to initialize any lazy components"""
        print("Warming up model...")
        try:
            dummy_audio = np.random.randn(16000).astype(np.float32)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, dummy_audio, 16000)
                temp_path = f.name
            
            try:
                with torch.no_grad():
                    _ = self.asr_model.transcribe([temp_path], batch_size=1)
                print("Warmup completed successfully")
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"Warmup failed (non-critical): {e}")
        
    def execute(self, requests):
        """Execute inference on a batch of requests"""
        all_audio_paths = []
        
        for request in requests:
            audio_path_tensor = pb_utils.get_input_tensor_by_name(request, "audio_path")
            audio_path_np = audio_path_tensor.as_numpy()
            
            if audio_path_np.ndim == 2:
                audio_path = audio_path_np[0, 0]
            else:
                audio_path = audio_path_np[0]
            
            if isinstance(audio_path, bytes):
                audio_path = audio_path.decode('utf-8')
            
            full_audio_path = os.path.join("/data/uploads", audio_path)
            all_audio_paths.append(full_audio_path)
        
        temp_files = []
        processed_paths = []
        
        try:
            for audio_path in all_audio_paths:
                audio, sr = librosa.load(audio_path, sr=None)
                
                if sr != self.target_sr:
                    audio_resampled = librosa.resample(
                        audio, 
                        orig_sr=sr, 
                        target_sr=self.target_sr
                    )
                else:
                    audio_resampled = audio
                
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.wav', 
                    delete=False
                )
                temp_path = temp_file.name
                temp_file.close()
                
                sf.write(temp_path, audio_resampled, self.target_sr)
                temp_files.append(temp_path)
                processed_paths.append(temp_path)
            
            with torch.no_grad():
                transcriptions = self.asr_model.transcribe(
                    processed_paths,
                    num_workers=4,
                    batch_size=8,
                    timestamps=True,
                    return_hypotheses=True
                )
            
        finally:
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        responses = []
        
        for idx, request in enumerate(requests):
            result = transcriptions[idx]
            transcription_text = result.text
            
            word_timestamps = []
            if hasattr(result, 'timestamp') and result.timestamp:
                if isinstance(result.timestamp, dict):
                    word_timestamps = result.timestamp.get("word", [])
                else:
                    word_timestamps = result.timestamp
            
            output_data = {
                'text': transcription_text,
                'word_timestamps': word_timestamps
            }
            
            output_json = json.dumps(output_data)
            transcription_array = np.array([[output_json]], dtype=object)
            
            output_tensor = pb_utils.Tensor("transcription", transcription_array)
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
        
        
        torch.cuda.empty_cache()
        return responses
    
    def finalize(self):
        """Clean up resources"""
        del self.asr_model
        torch.cuda.empty_cache()