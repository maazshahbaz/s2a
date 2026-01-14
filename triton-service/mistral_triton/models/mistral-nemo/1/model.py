import json
import os
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner, SamplingConfig


class TritonPythonModel:
    """
    Memory-optimized Mistral-Nemo model for H100 with single-request processing
    """

    def initialize(self, args):
        """
        Initialize tokenizer and TensorRT-LLM engine with H100 optimizations
        """
        self.model_config = json.loads(args['model_config'])
        model_repository = args['model_repository']
        
        self.model_base_dir = f"{model_repository}"
        self.tokenizer_dir = os.path.join(self.model_base_dir, "mistral-nemo-instruct-2407")
        self.engine_dir = os.path.join(self.model_base_dir, "engine")
        
        if not os.path.exists(self.tokenizer_dir):
            raise FileNotFoundError(f"Tokenizer directory not found: {self.tokenizer_dir}")
        
        if not os.path.exists(self.engine_dir):
            raise FileNotFoundError(f"Engine directory not found: {self.engine_dir}")
        
        parameters = self.model_config.get('parameters', {})
        self.max_tokens = int(parameters.get('max_tokens', {}).get('string_value', '800'))
        self.temperature = float(parameters.get('temperature', {}).get('string_value', '0.1'))
        self.top_k = int(parameters.get('top_k', {}).get('string_value', '20'))
        self.top_p = float(parameters.get('top_p', {}).get('string_value', '0.9'))
        self.repetition_penalty = float(parameters.get('repetition_penalty', {}).get('string_value', '1.1'))
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            trust_remote_code=True,
            padding_side='left',
            use_fast=True,
            model_max_length=8192
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.runner = ModelRunner.from_dir(
            engine_dir=self.engine_dir,
            rank=0,
            # max_batch_size=1,
            # max_input_len=8192,
            max_output_len=self.max_tokens,
            # max_beam_width=1,
            # kv_cache_enable_block_reuse=True,
            # kv_cache_free_gpu_memory_fraction=0.2,
            # enable_chunked_context=True
        )
        
        self.sampling_config = SamplingConfig(
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            # output_cum_log_probs=False,
            # output_log_probs=False,
            # beam_width=1
        )

        self.max_safe_input = 8192 - self.max_tokens - 100

    def _extract_prompt(self, request):
        """Extract prompt from a single request"""
        text_input = pb_utils.get_input_tensor_by_name(request, "prompt")
        prompt_np = text_input.as_numpy().flatten()
        prompt = prompt_np[0]
        
        if isinstance(prompt, bytes):
            prompt = prompt.decode('utf-8')
        
        return prompt

    def execute(self, requests):
        """
        Execute inference for single requests with memory-efficient processing
        """
        responses = []
        
        for request in requests:
            try:
                prompt = self._extract_prompt(request)
                
                encoded = self.tokenizer(
                    prompt,
                    padding=False,
                    truncation=True,
                    max_length=self.max_safe_input,
                    return_tensors="pt",
                    return_attention_mask=True
                )
                
                input_ids = encoded['input_ids']
                print(f"input_ids: {input_ids.shape[1]}", flush = True)
                input_length = encoded['attention_mask'].sum().item()
                
                batch_input_ids = [input_ids[0]]
                
                with torch.cuda.amp.autocast(enabled=False):
                    outputs = self.runner.generate(
                        batch_input_ids=batch_input_ids,
                        sampling_config=self.sampling_config
                    )
                
                output_ids = outputs[0, 0]
                generated_ids = output_ids[input_length:]
                
                if torch.is_tensor(generated_ids):
                    generated_ids = generated_ids.cpu().tolist()
                
                output_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                
                output_tensor = pb_utils.Tensor(
                    "generated_text",
                    np.array([output_text.encode('utf-8')], dtype=object)
                )
                
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(inference_response)
                
                del encoded, input_ids, outputs, output_ids, generated_ids
                
            except Exception as e:
                error_message = f"Error during inference: {str(e)}"
                
                error_tensor = pb_utils.Tensor(
                    "generated_text",
                    np.array([error_message.encode('utf-8')], dtype=object)
                )
                
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[error_tensor],
                    error=pb_utils.TritonError(error_message)
                )
                responses.append(inference_response)
            
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return responses

    def finalize(self):
        """Cleanup resources"""
        try:
            del self.runner
            del self.tokenizer
            del self.sampling_config
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            pass