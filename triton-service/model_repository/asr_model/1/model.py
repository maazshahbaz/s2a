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

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model"""
        self.model_config = json.loads(args['model_config'])
        
        # Load the ASR model once during initialization
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )
        self.target_sr = 16000
        
    def execute(self, requests):
        """Execute inference on a batch of requests"""
        # Collect all audio paths from all requests (1 per request)
        all_audio_paths = []
        
        for request in requests:
            audio_path_tensor = pb_utils.get_input_tensor_by_name(request, "audio_path")
            audio_path_np = audio_path_tensor.as_numpy()
            
            # Handle both batched and non-batched cases
            # When batched: shape is (batch_size, 1)
            # When not batched: shape is (1,)
            if audio_path_np.ndim == 2:
                audio_path = audio_path_np[0, 0]
            else:
                audio_path = audio_path_np[0]
            
            # Decode bytes to string
            if isinstance(audio_path, bytes):
                audio_path = audio_path.decode('utf-8')
            
            full_audio_path = os.path.join("/data/uploads", audio_path)
            all_audio_paths.append(full_audio_path)
        
        # Process all audio files at once
        temp_files = []
        processed_paths = []
        
        try:
            # Preprocess all audio files from all requests
            for audio_path in all_audio_paths:
                # Load and resample audio
                audio, sr = librosa.load(audio_path, sr=None)
                
                if sr != self.target_sr:
                    audio_resampled = librosa.resample(
                        audio, 
                        orig_sr=sr, 
                        target_sr=self.target_sr
                    )
                else:
                    audio_resampled = audio
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.wav', 
                    delete=False
                )
                temp_path = temp_file.name
                temp_file.close()
                
                sf.write(temp_path, audio_resampled, self.target_sr)
                temp_files.append(temp_path)
                processed_paths.append(temp_path)
            
            # Transcribe ALL audio files at once (single inference call)
            transcriptions = self.asr_model.transcribe(
                processed_paths,
                # processed_paths,
                num_workers=8
            )
            
        finally:
            # Clean up temporary files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        # Create responses (1 transcription per request)
        responses = []
        
        for idx, request in enumerate(requests):
            transcription_text = transcriptions[idx].text
            
            # Convert to numpy array with proper shape
            transcription_array = np.array(
                [[transcription_text]], 
                dtype=object
            )
            
            # Create output tensor
            output_tensor = pb_utils.Tensor(
                "transcription",
                transcription_array
            )
            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)
        del transcriptions
        gc.collect()

        self.asr_model.disable_cuda_graphs()

        torch.cuda.empty_cache()


        return responses
    
    def finalize(self):
        """Clean up resources"""
        del self.asr_model