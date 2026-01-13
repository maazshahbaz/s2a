import tempfile
import os
import tritonclient.grpc.aio as grpcclient_aio
import numpy as np
import soundfile as sf
from typing import Dict
import json
import librosa
from scipy import signal
from config_loader import config

class AsyncDiarizationClient:
    """Async gRPC Client for Triton Diarization Server"""
    
    def __init__(self, url: str = None):
        # Load configuration
        service_config = config.get_service_config('diarization')
        
        self.url = url or service_config.get('url', 'localhost:3601')
        self.model_name = service_config.get('model_name', 'diar')
        self.client = None
    
    async def connect(self):
        """Connect to Triton server"""
        self.client = grpcclient_aio.InferenceServerClient(
            url=self.url
        )
        
        if not await self.client.is_server_live():
            raise Exception(f"Triton server at {self.url} is not live")
        
        if not await self.client.is_server_ready():
            raise Exception(f"Triton server at {self.url} is not ready")
        
        if not await self.client.is_model_ready(self.model_name):
            raise Exception(f"Model {self.model_name} is not ready")

    
    def load_audio(self, audio_path: str, target_sr: int = None) -> tuple:
        """
        Load audio file, optionally resample to target sample rate.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (None = no resampling)
            
        Returns:
            (audio_data, sample_rate) tuple
        """
        audio_data, sample_rate = sf.read(audio_path)
        
        # Convert to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if requested
        if target_sr and sample_rate != target_sr:
            print(f"[Diarization] Resampling audio from {sample_rate} Hz to {target_sr} Hz")
            
            # Try using librosa (best quality)
            try:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=target_sr,
                    res_type='kaiser_best'
                )
            except ImportError:
                # Fallback to scipy
                
                num_samples = int(len(audio_data) * target_sr / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                
                # Ensure 1D array
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()
            
            sample_rate = target_sr
        
        # Ensure proper format for Triton
        audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)
        
        return audio_data, sample_rate
    
    async def diarize(self, audio_path: str, request_id: str) -> Dict:
        """Perform diarization on a single audio file"""
        
        # Check if resampling is needed
        audio_info = sf.info(audio_path)
        
        if audio_info.samplerate != 16000:
            print(f"[Diarization] Resampling from {audio_info.samplerate} Hz to 16000 Hz")
            
            # Load and resample
            audio_data, sample_rate = sf.read(audio_path)
            
            # Convert to mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample using librosa or scipy
            try:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000,
                    res_type='kaiser_best'
                )
            except Exception as e:
                print(f"Error resampling the audio file using librosa: {e}")
                print("using scipy to resample")
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
                sf.write(temp_path, audio_data, 16000, subtype='PCM_16')
            
            # Use temp file for diarization
            try:
                result = await self._diarize_internal(temp_path, request_id)
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            return result
        else:
            # Already 16 KHz, use directly
            return await self._diarize_internal(audio_path, request_id)
    
    async def _diarize_internal(self, audio_path: str, request_id: str) -> Dict:
        """Internal method to perform diarization (assumes audio is already 16 KHz)"""
        # Load audio
        audio_data, sample_rate = self.load_audio(audio_path, target_sr=None)  # No resampling
        
        # Create input tensors
        audio_input = grpcclient_aio.InferInput(
            "audio_input",
            [1, audio_data.shape[0]],
            "FP32"
        )
        audio_input.set_data_from_numpy(audio_data.reshape(1, -1))
        
        sample_rate_input = grpcclient_aio.InferInput(
            "sample_rate",
            [1, 1],
            "INT32"
        )
        sample_rate_input.set_data_from_numpy(
            np.array([[sample_rate]], dtype=np.int32)
        )
        
        # Create output
        output = grpcclient_aio.InferRequestedOutput("diarization_output")
        
        # Send async request
        response = await self.client.infer(
            model_name=self.model_name,
            inputs=[audio_input, sample_rate_input],
            outputs=[output],
            request_id = request_id
        )
        
        # Parse response
        result_json = response.as_numpy("diarization_output")[0].decode('utf-8')
        result = json.loads(result_json)
        return result
    
    async def close(self):
        """Close client connection"""
        if self.client:
            await self.client.close()