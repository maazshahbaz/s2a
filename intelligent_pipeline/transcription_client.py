import tritonclient.grpc.aio as grpcclient_aio
from typing import Dict
import uuid
import numpy as np
import json
from config_loader import config

class AsyncTranscriptionService:
    def __init__(self, url: str = None):
        # Load configuration
        service_config = config.get_service_config('transcription')
        
        self.url = url or service_config.get('url', 'localhost:3501')
        self.model_name = service_config.get('model_name', 'asr_opt')
        self.triton_client = None
        
    async def initialize(self):
        """Initialize the async Triton client."""
        if self.triton_client is None:
            self.triton_client = grpcclient_aio.InferenceServerClient(url=self.url)
        
    async def transcribe_async(self, audio_file_path, request_id:str) -> Dict:
        """
        Transcribe audio and return text with word timestamps
        
        Returns:
            {'text': '...', 'word_timestamps': [{'text': '...', 'start': 0.0, 'end': 0.0}, ...]}
        """
        await self.initialize()
        
        # Prepare input
        audio_path_np = np.array([[audio_file_path]], dtype=object)
        
        inputs = [
            grpcclient_aio.InferInput("audio_path", [1, 1], "BYTES")
        ]
        inputs[0].set_data_from_numpy(audio_path_np)
        
        outputs = [
            grpcclient_aio.InferRequestedOutput("transcription")
        ]
        
        # Send request
        response = await self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id
        )
        
        
        transcription = response.as_numpy("transcription")[0][0]
        if isinstance(transcription, bytes):
            transcription = transcription.decode('utf-8')
        
        # Parse JSON response
        try:
            result = json.loads(transcription)
            # Ensure result has the expected structure
            if isinstance(result, dict):
                # If it has text but no word_timestamps, initialize empty list
                if 'text' in result and 'word_timestamps' not in result:
                    result['word_timestamps'] = []
                # If word_timestamps exists, ensure it's a list
                elif 'word_timestamps' in result and not isinstance(result['word_timestamps'], list):
                    result['word_timestamps'] = []

                return result
            else:
                # If result is not a dict, treat as plain text
                return {
                    'text': str(result),
                    'word_timestamps': []
                }
        except json.JSONDecodeError:
            # Fallback: treat as plain text
            return {
                'text': transcription,
                'word_timestamps': []
            }