import json
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Python backend model for Llama 3.1 8B Instruct
    """
    
    def initialize(self, args):
        """
        Initialize the model on server startup
        
        Args:
            args: dict containing model configuration
        """
        # Parse model config
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get configuration parameters
        params = model_config.get('parameters', {})
        self.max_tokens = int(self._get_param(params, 'default_max_tokens', '512'))
        self.temperature = float(self._get_param(params, 'default_temperature', '0.7'))
        self.top_p = float(self._get_param(params, 'default_top_p', '0.9'))
        self.top_k = int(self._get_param(params, 'default_top_k', '50'))
        
        # Model configuration
        model_name = self._get_param(params, 'model_name', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
        use_flash_attention = self._get_param(params, 'use_flash_attention', 'false').lower() == 'true'
        load_in_8bit = self._get_param(params, 'load_in_8bit', 'false').lower() == 'true'
        
        # Set cache directory
        cache_dir = os.getenv('TRANSFORMERS_CACHE', '/models/.cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # HuggingFace token
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable must be set")
        
        print(f"Loading model: {model_name}")
        print(f"Cache directory: {cache_dir}")
        print(f"Flash Attention: {use_flash_attention}")
        print(f"8-bit quantization: {load_in_8bit}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            cache_dir=cache_dir,
            use_fast=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            'token': hf_token,
            'cache_dir': cache_dir,
            'torch_dtype': torch.float16,
            'device_map': 'auto',
        }
        
        if use_flash_attention:
            try:
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                print("Enabling Flash Attention 2")
            except Exception as e:
                print(f"Flash Attention not available: {e}")
        
        if load_in_8bit:
            model_kwargs['load_in_8bit'] = True
            print("Loading in 8-bit mode")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Enable better transformer optimizations
        try:
            # PyTorch 2.0+ optimization
            if hasattr(torch, 'compile') and not load_in_8bit:
                print("Compiling model with torch.compile()...")
                self.model = torch.compile(self.model)
        except Exception as e:
            print(f"Could not compile model: {e}")
        
        print("Model loaded successfully!")
        
        # Get device
        self.device = next(self.model.parameters()).device
        print(f"Model device: {self.device}")
    
    def _get_param(self, params, key, default):
        """Helper to get parameter from config"""
        if key in params:
            return params[key]['string_value']
        return default
    
    def _extract_string(self, tensor_data):
        """
        Extract string from tensor data handling various formats
        """
        if isinstance(tensor_data, bytes):
            return tensor_data.decode('utf-8')
        elif isinstance(tensor_data, np.ndarray):
            # Handle numpy array
            if tensor_data.size == 0:
                return ""
            # Get first element
            item = tensor_data.flatten()[0]
            if isinstance(item, bytes):
                return item.decode('utf-8')
            elif isinstance(item, str):
                return item
            else:
                return str(item)
        elif isinstance(tensor_data, str):
            return tensor_data
        else:
            return str(tensor_data)
    
    def execute(self, requests):
        """
        Execute inference on a batch of requests
        
        Args:
            requests: list of pb_utils.InferenceRequest
            
        Returns:
            list of pb_utils.InferenceResponse
        """
        responses = []
        
        for request in requests:
            try:
                # Get input tensors
                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
                prompt_np = prompt_tensor.as_numpy()
                prompt = self._extract_string(prompt_np)
                
                # Get optional parameters
                max_tokens = self._get_request_param(request, "max_tokens", self.max_tokens)
                temperature = self._get_request_param(request, "temperature", self.temperature)
                top_p = self._get_request_param(request, "top_p", self.top_p)
                top_k = self._get_request_param(request, "top_k", self.top_k)
                
                # Get system prompt
                system_prompt_tensor = pb_utils.get_input_tensor_by_name(request, "system_prompt")
                if system_prompt_tensor is not None:
                    system_prompt_np = system_prompt_tensor.as_numpy()
                    system_prompt = self._extract_string(system_prompt_np)
                else:
                    system_prompt = "You are a helpful AI assistant."
                
                print(f"Processing request - Prompt length: {len(prompt)}, Max tokens: {max_tokens}")
                
                # Format as chat if needed
                if system_prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    formatted_prompt = prompt
                
                # Tokenize
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096
                ).to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode output
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                print(f"Generated response length: {len(generated_text)}")
                
                # Create output tensor - ensure proper format for batching
                output_data = np.array([[generated_text.encode('utf-8')]], dtype=np.object_)
                output_tensor = pb_utils.Tensor(
                    "generated_text",
                    output_data
                )
                
                # Create response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(inference_response)
                
            except Exception as e:
                # Handle errors
                error_msg = f"Error during inference: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_msg)
                )
                responses.append(error_response)
        
        return responses
    
    def _get_request_param(self, request, name, default):
        """Get optional parameter from request"""
        try:
            tensor = pb_utils.get_input_tensor_by_name(request, name)
            if tensor is None:
                return default
            
            value_np = tensor.as_numpy()
            
            # Handle different data types and shapes
            if value_np.size == 0:
                return default
            
            # Flatten and get first element
            value = value_np.flatten()[0]
            
            # Handle different data types
            if isinstance(value, bytes):
                return value.decode('utf-8')
            elif isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            else:
                return value
        except Exception as e:
            print(f"Error getting parameter {name}: {e}")
            return default
    
    def finalize(self):
        """
        Clean up resources on server shutdown
        """
        print("Cleaning up model resources...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Model finalized successfully")
