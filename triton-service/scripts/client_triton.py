"""
gRPC Client to send call transcription analysis requests to Llama 3.1 on Triton
"""
import json
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class TritonClient:
    def __init__(self, url, model_name):
        """Initialize Triton gRPC client for call analysis"""
        self.url = url
        self.model_name = model_name
        self.client = grpcclient.InferenceServerClient(url=url)

    def analyze_call(self, transcription, max_tokens=512, temperature=0.3, 
                     top_p=0.9):
        """
        Analyze call transcription and return structured JSON
        
        Args:
            transcription: Call transcription text
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.3 for more focused output)
            top_p: Nucleus sampling (default: 0.9)
            verbose: Print details
            
        Returns:
            dict: Parsed JSON response with call analysis
        """

        # Construct the prompt
        prompt = f"""                                                                                                                        You are an expert call center quality analyst. The following text is a call transcription written as a single paragraph with no speaker labels.
    Your task is to infer the likely dialogue flow between the customer and the agent, and then provide a structured analysis of the call.

    Call Transcription:
    {transcription}

    Perform the following steps:
    1. Reconstruct the conversation in your mind by inferring which parts are spoken by the customer and which by the agent.
    2. Analyze the inferred conversation for tone, resolution, and key insights.

    Respond ONLY with a valid JSON object (no extra text or explanations).

    The JSON must strictly follow this structure:
    {{
        "call_sentiment": "positive" | "negative" | "neutral",
        "call_summary": "A summary of the main points of the call.",
        "call_status": "resolved" | "pending" | "escalated",
        "call_improvement_points": [
            "Specific, actionable improvement point 1",
            "Specific, actionable improvement point 2"
        ],
        "key_words": [
            "keyword1",
            "keyword2",
            "keyword3"
        ]
    }}
    """


        # System prompt
        system_prompt = "You are a helpful assistant that analyzes call center transcriptions and provides structured insights."

        try:
            # Prepare inputs with proper batch dimensions
            inputs = []

            # Main prompt (BYTES type) - shape [1, 1] for batch size
            prompt_data = np.array([[prompt.encode('utf-8')]], dtype=np.object_)
            input_prompt = grpcclient.InferInput("prompt", [1, 1], "BYTES")
            input_prompt.set_data_from_numpy(prompt_data)
            inputs.append(input_prompt)

            # System prompt (BYTES type) - shape [1, 1] for batch size
            system_data = np.array([[system_prompt.encode('utf-8')]], dtype=np.object_)
            input_system = grpcclient.InferInput("system_prompt", [1, 1], "BYTES")
            input_system.set_data_from_numpy(system_data)
            inputs.append(input_system)

            # Max tokens (INT32 type) - shape [1, 1] for batch size
            max_tokens_data = np.array([[max_tokens]], dtype=np.int32)
            input_max_tokens = grpcclient.InferInput("max_tokens", [1, 1], "INT32")
            input_max_tokens.set_data_from_numpy(max_tokens_data)
            inputs.append(input_max_tokens)

            # Temperature (FP32 type) - shape [1, 1] for batch size
            temp_data = np.array([[temperature]], dtype=np.float32)
            input_temp = grpcclient.InferInput("temperature", [1, 1], "FP32")
            input_temp.set_data_from_numpy(temp_data)
            inputs.append(input_temp)

            # Top-p (FP32 type) - shape [1, 1] for batch size
            top_p_data = np.array([[top_p]], dtype=np.float32)
            input_top_p = grpcclient.InferInput("top_p", [1, 1], "FP32")
            input_top_p.set_data_from_numpy(top_p_data)
            inputs.append(input_top_p)

            # Prepare outputs
            outputs = [
                grpcclient.InferRequestedOutput("generated_text")
            ]

            # Run inference
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )

            # Get result
            generated_text = response.as_numpy("generated_text")[0]
            if isinstance(generated_text, bytes):
                generated_text = generated_text.decode('utf-8')
            elif isinstance(generated_text, np.ndarray):
                generated_text = generated_text[0].decode('utf-8') if isinstance(generated_text[0], bytes) else str(generated_text[0])

            # Parse JSON from response
            try:
                # Try to extract JSON if there's extra text
                json_start = generated_text.find('{')
                json_end = generated_text.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = generated_text[json_start:json_end]
                    result = json.loads(json_str)

                    return result
                else:

                    return {"error": "No valid JSON in response", "raw_response": generated_text}

            except json.JSONDecodeError as e:
                print(e)
                return {"error": "Invalid JSON", "raw_response": generated_text}

        except InferenceServerException as e:
            return {"error": str(e)}

        except Exception as e:
            return {"error": str(e)}
            
