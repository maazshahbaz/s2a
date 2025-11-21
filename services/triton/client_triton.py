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

    def analyze_call(self, transcription, max_tokens=1024, temperature=0.3,
                     top_p=0.9):
        """
        Analyze call transcription and return structured JSON

        Args:
            transcription: Call transcription text
            max_tokens: Maximum tokens to generate (default: 1024 for comprehensive output)
            temperature: Sampling temperature (default: 0.3 for more structured output)
            top_p: Nucleus sampling (default: 0.9)

        Returns:
            dict: Comprehensive parsed JSON response with call analysis
        """

        # Construct the enhanced prompt for comprehensive analysis
        prompt = f"""You are an expert AI system specializing in call center analytics, fraud detection, and business intelligence extraction.

Analyze the following call transcription and provide a comprehensive structured analysis.

Call Transcription:
{transcription}

Provide a detailed analysis following this EXACT JSON structure. Be thorough and specific:

{{
    "ai_analysis": {{
        "call_type": {{
            "human_to_human": true/false,
            "ivr": true/false,
            "voicemail": true/false
        }},
        "sentiment": {{
            "category": "Positive|Negative|Neutral|Mixed",
            "confidence": 0.0-1.0,
            "key_indicators": ["list of phrases that indicate the sentiment"]
        }},
        "summary": "Detailed summary with key outcomes and decisions",
        "call_status": {{
            "voicemail_message_request": true/false,
            "do_not_disturb": true/false,
            "wrong_number": true/false,
            "callback_requested": true/false
        }},
        "extracted_items": {{
            "products": [
                {{
                    "name": "product/service name",
                    "quantity": number or null,
                    "mentioned_at": "context or sequence reference"
                }}
            ],
            "action_items": [
                {{
                    "description": "what needs to be done",
                    "owner": "who is responsible",
                    "deadline": "when it's due or null"
                }}
            ],
            "contact_info": {{
                "phone": "phone number or null",
                "email": "email address or null",
                "name": "person's name or null"
            }}
        }},
        "fraud_detection": {{
            "suspicious_language": true/false,
            "high_pressure_tactics": true/false,
            "risk_level": "Low|Medium|High",
            "evidence": ["specific phrases or patterns that indicate risk"]
        }}
    }}
}}

IMPORTANT:
- Return ONLY valid JSON with no additional text
- Use null for missing values, never omit fields
- Ensure all boolean values are lowercase (true/false)
- Keep arrays empty [] if no items found
- Be specific and detailed in descriptions"""


        # Enhanced system prompt for comprehensive analysis
        system_prompt = """You are an expert AI system specializing in call center analytics with advanced capabilities in:
- Fraud detection and risk assessment
- Sentiment analysis with confidence scoring
- Entity and information extraction
- Business intelligence and opportunity identification
- Call quality assessment and improvement recommendations
Your responses must be precise, structured JSON that captures both high-level insights and granular details."""

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
            
