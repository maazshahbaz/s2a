import tritonclient.grpc as grpcclient
import numpy as np
import json
import uuid
from typing import Optional
import time


class ASRTritonClient:
    def __init__(self, triton_url: str = "localhost:2001", model_name: str = "asr_model"):
        """
        Initialize Triton gRPC client for ASR
        
        Args:
            triton_url: Triton server URL (gRPC port)
            model_name: Model name in Triton
        """
        self.model_name = model_name
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)

        
        # Check if model is loaded
        if not self.triton_client.is_model_ready(self.model_name):
            raise Exception(f"Model {self.model_name} is not ready")
        
        print(f"✅ Connected to Triton gRPC server. Model {self.model_name} is ready.")
    
    def _callback(self, result, error):
        """
        Callback function for async inference - prints result as JSON
        
        Args:
            result: InferResult object
            error: Error object if request failed
        """
        request_id = result.get_response().id
        print(request_id)
        
        if error:
            print(json.dumps({
                "request_id": request_id,
                "success": False,
                "error": str(error)
            }, indent=2))
        else:
            # Get transcription
            # print(result.as_numpy("transcription")[0][0])
            transcription = result.as_numpy("transcription")[0][0]
            
            # Decode if bytes
            if isinstance(transcription, bytes):
                transcription = transcription.decode('utf-8')
            
            print(json.dumps({
                "request_id": request_id,
                "success": True,
                "transcription": transcription
            }, indent=2))
    
    def transcribe_async(self, audio_file_path: str, request_id: Optional[str] = None, on_complete=None) -> str:
        """
        Transcribe a single audio file asynchronously
        
        Args:
            audio_file_path: Relative path to audio file (e.g., "2025-11-18/audio.wav")
            request_id: Optional request ID (will be generated if not provided)
            
        Returns:
            request_id: The request ID
        """
        # Generate request_id if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Prepare input data (just the relative path)
        audio_path_np = np.array([[audio_file_path]], dtype=object)
        
        # Create input tensor
        inputs = [
            grpcclient.InferInput("audio_path", [1, 1], "BYTES")
        ]
        inputs[0].set_data_from_numpy(audio_path_np)
        
        # Create output placeholder
        outputs = [
            grpcclient.InferRequestedOutput("transcription")
        ]
        
        # Send async inference request with callback
        self.triton_client.async_infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            callback=on_complete,
            request_id=request_id
        )
        
        print(f"📤 Sent async transcription request: {request_id}")
        return request_id
    
    def transcribe_batch_async(self, audio_file_paths: list[str]) -> list[str]:
        """
        Transcribe multiple audio files asynchronously
        
        Args:
            audio_file_paths: List of relative paths (max 16 due to server config)
            
        Returns:
            List of request_ids
        """
        if len(audio_file_paths) > 16:
            raise ValueError("Batch size cannot exceed 16")
        
        request_ids = []
        
        # Send all requests asynchronously
        for audio_path in audio_file_paths:
            request_id = self.transcribe_async(audio_path)
            request_ids.append(request_id)
        
        return request_ids
    
    def close(self):
        """Close the client connection"""
        self.triton_client.close()


# Example usage
if __name__ == "__main__":
    client = ASRTritonClient(triton_url="localhost:2001")
    
    print("\n" + "="*80)
    print("Single async transcription")
    print("="*80)
    
    # Single file - result will be printed by callback
    request_id = client.transcribe_async("2025-11-18/fe5cf860-62c3-45a0-8782-35fe0a482beb.wav")
    
    # Wait for callback to complete
    time.sleep(20)
    
    print("\n" + "="*80)
    print("Batch async transcription")
    print("="*80)
    
    # Multiple files
    audio_files = [
        "2025-11-18/fe5cf860-62c3-45a0-8782-35fe0a482beb.wav",
        "2025-11-18/fe5cf860-62c3-45a0-8782-35fe0a482beb.wav",
        "2025-11-18/fe5cf860-62c3-45a0-8782-35fe0a482beb.wav"
    ]
    
    request_ids = client.transcribe_batch_async(audio_files)
    print(f"Sent {len(request_ids)} transcription requests")
    
    # Wait for all callbacks to complete
    print("\nWaiting for results (callbacks will print automatically)...")
    time.sleep(60)
    
    # Close connection
    client.close()
    print("\n✅ Client closed")