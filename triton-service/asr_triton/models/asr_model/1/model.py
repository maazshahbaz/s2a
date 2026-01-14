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
        """Initialize the model with H100 optimizations"""
        self.model_config = json.loads(args['model_config'])
        
        print("Loading ASR model...")
        # Load the ASR model
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )
        
        # Move to GPU
        self.asr_model = self.asr_model.cuda()
        
        # H100 Optimization: Use bfloat16 for better performance
        if torch.cuda.is_bf16_supported():
            self.asr_model = self.asr_model.to(dtype=torch.bfloat16)
            self.amp_dtype = torch.bfloat16
            print("Using bfloat16 precision for H100")
        else:
            self.amp_dtype = torch.float16
            print("Using float16 precision")
        
        self.asr_model.eval()
        
        # H100 Optimization: Enable Flash Attention 2
        self._enable_flash_attention()
        
        self.target_sr = 16000
        
        # Enable decoding optimizations (disable CUDA graphs for TDT compatibility)
        self._enable_optimizations()
        
        # H100 Optimization: Compile model with torch.compile
        self._compile_model()
        
        # Warm up the model
        self._warmup()
        
        print("Model initialization complete")
    
    def _enable_flash_attention(self):
        """Enable Flash Attention 2 for H100"""
        try:
            # Enable Flash Attention through PyTorch backends
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                print("Enabled Flash Attention 2")
            
            # Set memory format for better performance
            if hasattr(self.asr_model, 'encoder'):
                try:
                    self.asr_model.encoder = self.asr_model.encoder.to(
                        memory_format=torch.channels_last
                    )
                    print("Set channels_last memory format")
                except Exception as e:
                    print(f"Could not set memory format: {e}")
                    
        except Exception as e:
            print(f"Flash Attention setup failed (non-critical): {e}")
    
    def _compile_model(self):
        """Compile model for H100 using torch.compile"""
        try:
            print("Compiling model with torch.compile...")
            self.asr_model = torch.compile(
                self.asr_model,
                mode='reduce-overhead',  # Options: 'default', 'reduce-overhead', 'max-autotune'
                fullgraph=False,
                dynamic=True
            )
            print("Model compiled successfully")
        except Exception as e:
            print(f"Model compilation failed (will use uncompiled): {e}")
    
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
            # Create dummy audio
            dummy_audio = np.random.randn(self.target_sr).astype(np.float32)
            
            # Warm up with actual transcription
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.amp_dtype):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    sf.write(f.name, dummy_audio, self.target_sr)
                    temp_path = f.name
                
                try:
                    _ = self.asr_model.transcribe(
                        [temp_path], 
                        batch_size=1,
                        num_workers=0
                    )
                    print("Warmup completed successfully")
                finally:
                    os.unlink(temp_path)
                    
        except Exception as e:
            print(f"Warmup failed (non-critical): {e}")
    
    def _extract_path(self, audio_path_tensor):
        """Extract audio path from tensor"""
        audio_path_np = audio_path_tensor.as_numpy()
        
        if audio_path_np.ndim == 2:
            audio_path = audio_path_np[0, 0]
        else:
            audio_path = audio_path_np[0]
        
        if isinstance(audio_path, bytes):
            audio_path = audio_path.decode('utf-8')
        
        return audio_path
    
    def execute(self, requests):
        """Execute inference on a batch of requests with optimized processing"""
        if not requests:
            return []
        
        all_audio_paths = []
        audio_arrays = []
        
        try:
            # Extract all audio paths
            for request in requests:
                audio_path_tensor = pb_utils.get_input_tensor_by_name(request, "audio_path")
                audio_path = self._extract_path(audio_path_tensor)
                full_audio_path = os.path.join("/data/uploads", audio_path)
                all_audio_paths.append(full_audio_path)
            
            # Load and resample all audio files into memory
            # This avoids creating temporary files which is more efficient
            for audio_path in all_audio_paths:
                try:
                    # Load audio with target sample rate directly
                    # audio, sr = librosa.load(
                    #     audio_path, 
                    #     sr=self.target_sr,
                    #     mono=True
                    # )
                    audio_arrays.append(audio_path)
                    
                except Exception as e:
                    print(f"Error loading audio {audio_path}: {e}")
                    # Append empty audio on error
                    audio_arrays.append(np.zeros(self.target_sr, dtype=np.float32))
            
            # Perform batch transcription with AMP and all optimizations
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.amp_dtype):
                transcriptions = self.asr_model.transcribe(
                    audio_arrays,
                    num_workers=0,  # Disable multiprocessing in Triton (adds overhead)
                    batch_size=len(requests),  # Process entire batch at once
                    timestamps=True,
                    return_hypotheses=True
                )
            
            # Build responses
            responses = []
            for idx, request in enumerate(requests):
                try:
                    result = transcriptions[idx]
                    transcription_text = result.text
                    
                    # Extract word timestamps
                    word_timestamps = []
                    if hasattr(result, 'timestamp') and result.timestamp:
                        if isinstance(result.timestamp, dict):
                            word_timestamps = result.timestamp.get("word", [])
                        else:
                            word_timestamps = result.timestamp
                    
                    # Prepare output
                    output_data = {
                        'text': transcription_text,
                        'word_timestamps': word_timestamps
                    }
                    
                    output_json = json.dumps(output_data)
                    transcription_array = np.array([[output_json]], dtype=object)
                    
                    output_tensor = pb_utils.Tensor("transcription", transcription_array)
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[output_tensor]
                    )
                    responses.append(inference_response)
                    
                except Exception as e:
                    print(f"Error processing result {idx}: {e}")
                    # Return error response
                    error_data = {
                        'text': '',
                        'word_timestamps': [],
                        'error': str(e)
                    }
                    error_json = json.dumps(error_data)
                    error_array = np.array([[error_json]], dtype=object)
                    error_tensor = pb_utils.Tensor("transcription", error_array)
                    error_response = pb_utils.InferenceResponse(
                        output_tensors=[error_tensor]
                    )
                    responses.append(error_response)
            
            return responses
            
        except Exception as e:
            print(f"Batch processing error: {e}")
            # Return error responses for all requests
            responses = []
            for request in requests:
                error_data = {
                    'text': '',
                    'word_timestamps': [],
                    'error': str(e)
                }
                error_json = json.dumps(error_data)
                error_array = np.array([[error_json]], dtype=object)
                error_tensor = pb_utils.Tensor("transcription", error_array)
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[error_tensor]
                )
                responses.append(error_response)
            return responses
            
        finally:
            # Clean up memory
            del audio_arrays
            if 'transcriptions' in locals():
                del transcriptions
            torch.cuda.empty_cache()
    
    def finalize(self):
        """Clean up resources"""
        print("Finalizing model...")
        try:
            del self.asr_model
            torch.cuda.empty_cache()
            gc.collect()
            print("Model finalized successfully")
        except Exception as e:
            print(f"Error during finalization: {e}")