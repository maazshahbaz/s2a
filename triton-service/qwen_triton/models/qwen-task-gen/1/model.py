import json
import os
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class TritonPythonModel:
    """
    Qwen2.5-1.5B-Instruct model for agent task generation.
    Uses HuggingFace transformers with 8-bit quantization and H100 optimizations.
    """

    def initialize(self, args):
        """
        Initialize tokenizer and model with 8-bit quantization and H100 optimizations.
        """
        self.model_config = json.loads(args['model_config'])

        parameters = self.model_config.get('parameters', {})
        self.max_tokens = int(parameters.get('max_tokens', {}).get('string_value', '512'))
        self.temperature = float(parameters.get('temperature', {}).get('string_value', '0.1'))
        self.top_k = int(parameters.get('top_k', {}).get('string_value', '20'))
        self.top_p = float(parameters.get('top_p', {}).get('string_value', '0.9'))
        self.repetition_penalty = float(parameters.get('repetition_penalty', {}).get('string_value', '1.1'))

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        # H100 CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
            torch.set_grad_enabled(False)

        print(f"[Qwen Task Gen] Loading tokenizer from {model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left',
            use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"[Qwen Task Gen] Loading model in 8-bit from {model_name}", flush=True)

        # 8-bit quantization config (~1.5GB per instance)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        self.model.eval()

        # Warmup with a short prompt to initialize CUDA kernels
        self._warmup()

        print(f"[Qwen Task Gen] Model loaded successfully (8-bit quantized)", flush=True)

    def _warmup(self):
        """Run a short warmup inference to initialize CUDA kernels."""
        try:
            warmup_messages = [
                {"role": "system", "content": "You are a task generator."},
                {"role": "user", "content": "Hello"}
            ]
            warmup_text = self.tokenizer.apply_chat_template(
                warmup_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(warmup_text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=5)
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Qwen Task Gen] Warmup complete", flush=True)
        except Exception as e:
            print(f"[Qwen Task Gen] Warmup failed (non-fatal): {e}", flush=True)

    def execute(self, requests):
        """
        Execute inference for requests.
        """
        responses = []

        for request in requests:
            try:
                # Extract prompt
                text_input = pb_utils.get_input_tensor_by_name(request, "prompt")
                prompt_np = text_input.as_numpy().flatten()
                prompt = prompt_np[0]

                if isinstance(prompt, bytes):
                    prompt = prompt.decode('utf-8')

                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    padding=False,
                    truncation=True,
                    max_length=4096,
                    return_tensors="pt"
                ).to(self.model.device)

                input_length = inputs['input_ids'].shape[1]
                print(f"[Qwen Task Gen] input_ids: {input_length}", flush=True)

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                        repetition_penalty=self.repetition_penalty,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                # Decode only the generated tokens (exclude input)
                generated_ids = outputs[0][input_length:]
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

                del inputs, outputs, generated_ids

            except Exception as e:
                error_message = f"Error during inference: {str(e)}"
                print(f"[Qwen Task Gen] {error_message}", flush=True)

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
        """Cleanup resources."""
        try:
            del self.model
            del self.tokenizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            print(f"[Qwen Task Gen] Finalize error: {e}", flush=True)
