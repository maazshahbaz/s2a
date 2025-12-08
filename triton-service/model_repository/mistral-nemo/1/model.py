import json
import os
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner, SamplingConfig
import traceback


class TritonPythonModel:
    """
    Complete Mistral-Nemo model with tokenization, generation, and detokenization
    """

    def initialize(self, args):
        """
        Initialize tokenizer and TensorRT-LLM engine
        
        """
        
        # Parse model config from config.pbtxt
        self.model_config = json.loads(args['model_config'])
        
        # Get model name and repository from Triton
        model_name = args['model_name']  # "mistral-nemo"
        model_repository = args['model_repository']  # "/models"
        
        # Base model directory
        self.model_base_dir = f"{model_repository}"
        
        # Tokenizer directory (where the HF model is)
        self.tokenizer_dir = os.path.join(self.model_base_dir, "mistral-nemo-instruct-2407")
        
        # Engine directory (where TensorRT engine is)
        self.engine_dir = os.path.join(self.model_base_dir, "engine")
        
        # Log initialization info
        print("="*80)
        print("Triton Model Initialization")
        print("="*80)
        print(f"Model name: {args['model_name']}")
        print(f"Model instance: {args['model_instance_name']}")
        print(f"Model version: {args['model_version']}")
        print(f"Model repository: {args['model_repository']}")
        print(f"Model base directory: {self.model_base_dir}")
        print(f"Tokenizer directory: {self.tokenizer_dir}")
        print(f"Engine directory: {self.engine_dir}")
        print(f"Device: {args['model_instance_kind']} {args['model_instance_device_id']}")
        print("="*80)
        
        # Verify directories exist
        if not os.path.exists(self.tokenizer_dir):
            raise FileNotFoundError(
                f"Tokenizer directory not found: {self.tokenizer_dir}\n"
                f"Expected structure: {self.model_base_dir}/mistral-nemo-instruct-2407/"
            )
        
        if not os.path.exists(self.engine_dir):
            raise FileNotFoundError(
                f"Engine directory not found: {self.engine_dir}\n"
                f"Expected structure: {self.model_base_dir}/engine/"
            )
        
        # Get parameters from config.pbtxt
        parameters = self.model_config.get('parameters', {})
        self.max_tokens = int(
            parameters.get('max_tokens', {}).get('string_value', '800')
        )
        self.temperature = float(
            parameters.get('temperature', {}).get('string_value', '0.1')
        )
        self.top_k = int(
            parameters.get('top_k', {}).get('string_value', '20')
        )
        self.top_p = float(
            parameters.get('top_p', {}).get('string_value', '0.9')
        )
        self.repetition_penalty = float(
            parameters.get('repetition_penalty', {}).get('string_value', '1.1')
        )
        
        # Initialize tokenizer from mistral-nemo-instruct-2407 folder
        print(f"Loading tokenizer from {self.tokenizer_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"Tokenizer loaded successfully")
        print(f"   Vocab size: {len(self.tokenizer)}")
        print(f"   EOS token ID: {self.tokenizer.eos_token_id}")
        print(f"   PAD token ID: {self.tokenizer.pad_token_id}")
        
        # Initialize TensorRT-LLM engine from engine folder
        print(f"\nLoading TensorRT-LLM engine from {self.engine_dir}")
        
        # List engine files for debugging
        engine_files = os.listdir(self.engine_dir)
        print(f"   Engine files found: {engine_files}")
        
        self.runner = ModelRunner.from_dir(
            engine_dir=self.engine_dir,
            rank=0
        )
        print(f"TensorRT-LLM engine loaded successfully")
        
        # Configure sampling once
        self.sampling_config = SamplingConfig(
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
            # beam_width=1
        )
        
        print("\nModel Configuration:")
        print(f"   Max tokens: {self.max_tokens}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Top-k: {self.top_k}")
        print(f"   Top-p: {self.top_p}")
        print(f"   Repetition penalty: {self.repetition_penalty}")
        print("\nModel initialization complete!")
        print("="*80)

    def execute(self, requests):
        """
        Execute inference for a batch of requests WITHOUT loops
        All prompts are processed simultaneously
        """
        responses = []
        
        # Extract all prompts from requests at once
        prompts = []
        for request in requests:
            text_input = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt_np = text_input.as_numpy()
        
            # Handle both batched and non-batched cases
            # When batched: shape is (batch_size,) or (batch_size, 1)
            # When not batched: shape is (1,) or scalar
            if prompt_np.ndim == 0:
                # Scalar
                prompt = prompt_np.item()
            elif prompt_np.ndim == 1:
                # 1D array
                prompt = prompt_np[0]
            else:
                # 2D array (batch_size, 1)
                prompt = prompt_np[0, 0] if prompt_np.shape[1] == 1 else prompt_np[0]
            
            # Decode bytes to string if needed
            if isinstance(prompt, bytes):
                prompt = prompt.decode('utf-8')
            
            prompts.append(prompt)
        
        batch_size = len(prompts)
        print(f"\nProcessing batch of {batch_size} prompts")
        
        try:
            # Tokenize all prompts at once (batched tokenization)
            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            )
            
            # Get input_ids and attention_mask
            input_ids_batch = encoded['input_ids']
            input_lengths = (encoded['attention_mask'].sum(dim=1)).tolist()
            
            print(f"   Input IDs shape: {input_ids_batch.shape}")
            print(f"   Input lengths: {input_lengths}")
            
            # Prepare batch for TensorRT-LLM
            batch_input_ids = [input_ids_batch[i] for i in range(batch_size)]
            
            # Generate for entire batch at once
            outputs = self.runner.generate(
                batch_input_ids=batch_input_ids,
                sampling_config=self.sampling_config
            )
            
            # Process outputs
            print(outputs.shape)
            # output_ids_batch = outputs[0, 0]
            print(f"   Output IDs shape: {outputs.shape}")
            
            # Detokenize entire batch
            generated_texts = []
            for i in range(batch_size):
                output_ids = outputs[i, 0]
                input_length = input_lengths[i]
                generated_ids = output_ids[input_length:]
                
                if torch.is_tensor(generated_ids):
                    generated_ids = generated_ids.cpu().tolist()
                
                generated_texts.append(generated_ids)
            
            # Batch decode
            output_texts = self.tokenizer.batch_decode(
                generated_texts,
                skip_special_tokens=True
            )
            
            print(f"Generated {len(output_texts)} responses")
            
            # Create responses
            for output_text in output_texts:
                output_tensor = pb_utils.Tensor(
                    "generated_text",
                    np.array([output_text.encode('utf-8')], dtype=object)
                )
                
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(inference_response)
            
        except Exception as e:
            error_message = f"Error during batch inference: {str(e)}"
            print(f"{error_message}")
            traceback.print_exc()
            
            for _ in range(batch_size):
                error_tensor = pb_utils.Tensor(
                    "generated_text",
                    np.array([error_message.encode('utf-8')], dtype=object)
                )
                
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[error_tensor],
                    error=pb_utils.TritonError(error_message)
                )
                responses.append(inference_response)
        
        return responses

    def finalize(self):
        """Cleanup resources"""
        print('\nCleaning up model resources...')