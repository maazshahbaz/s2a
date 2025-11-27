import tritonclient.grpc.aio as grpcclient_aio
import json
import numpy as np
import uuid

class AsyncAnalysis:
    def __init__(self):
        self.model_name = "mistral-nemo"
        self.client = None
        
    async def initialize(self):
        """Initialize the async Triton client."""
        if self.client is None:
            self.client = grpcclient_aio.InferenceServerClient(url="localhost:2001")
    
    def __clean_output(self, output_text, request_id):
        """Clean and format the output."""
        try:
            output_text_cleaned = output_text.strip()
            if output_text_cleaned.startswith('```json'):
                output_text_cleaned = output_text_cleaned.replace('```json', '').replace('```', '').strip()
            elif output_text_cleaned.startswith('```'):
                output_text_cleaned = output_text_cleaned.replace('```', '').strip()
            
            start_idx = output_text_cleaned.find('{')
            end_idx = output_text_cleaned.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = output_text_cleaned[start_idx:end_idx]
                analysis = json.loads(json_str)
                
                return json.dumps({
                    "request_id": request_id,
                    "success": True,
                    "analysis": analysis
                }, indent=2)
            else:
                return json.dumps({
                    "request_id": request_id,
                    "success": True,
                    "generated_text": output_text
                }, indent=2)
                
        except json.JSONDecodeError:
            return json.dumps({
                "request_id": request_id,
                "success": True,
                "generated_text": output_text
            }, indent=2)
    
    async def __generate_async(self, prompt, request_id=None) -> str:
        """Generate text asynchronously."""
        await self.initialize()
        
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        input_data = np.array([[prompt]], dtype=object)
        inputs = [grpcclient_aio.InferInput("prompt", [1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient_aio.InferRequestedOutput("generated_text")]
        
        response = await self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id
        )
        
        output_text = response.as_numpy("generated_text")[0].decode('utf-8')
        
        if isinstance(output_text, bytes):
            output_text = output_text.decode('utf-8')
        
        json_output = self.__clean_output(output_text, request_id)
        return json_output
    
    def __format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format prompts according to Mistral's instruction format"""
        return f"""[INST] {system_prompt}

{user_prompt} [/INST]"""
    
    async def analyze_call_async(self, transcription: str, request_id=None) -> str:
        """
        Analyze call center transcription asynchronously.
        
        Args:
            transcription: RAW transcription text WITHOUT speaker labels
        """
        system_prompt = """You are an expert AI system specializing in call center analytics with advanced capabilities in:
- Fraud detection and risk assessment
- Sentiment analysis with confidence scoring
- Entity and information extraction
- Business intelligence and opportunity identification
- Call quality assessment and improvement recommendations
Your responses must be precise, structured JSON that captures both high-level insights and granular details."""
        
        user_prompt = f"""You are an expert AI system specializing in call center analytics, fraud detection, and business intelligence extraction.
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

IMPORTANT:
- Return ONLY valid JSON with no additional text
- Use null for missing values, never omit fields
- Ensure all boolean values are lowercase (true/false)
- Keep arrays empty [] if no items found
- Be specific and detailed in descriptions"""
    
        full_prompt = self.__format_prompt(system_prompt, user_prompt)
        return await self.__generate_async(full_prompt, request_id)

