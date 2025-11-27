import tritonclient.grpc.aio as grpcclient_aio
import numpy as np
import soundfile as sf
from typing import Dict
import json

class AsyncDiarizationClient:
    """Async gRPC Client for Triton Diarization Server"""
    
    def __init__(self, url: str = "localhost:2001"):
        self.url = url

        self.model_name = "diarization_model"
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

    
    def load_audio(self, audio_path: str) -> tuple:
        """Load audio file"""
        audio_data, sample_rate = sf.read(audio_path)
        
        # Convert to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Ensure float32
        audio_data = audio_data.astype(np.float32)
        
        return audio_data, sample_rate
    
    async def diarize(self, audio_path: str, request_id: str) -> Dict:
        """Perform diarization on a single audio file"""
        # Load audio
        audio_data, sample_rate = self.load_audio(audio_path)
        
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

