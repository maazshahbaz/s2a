import tritonclient.grpc.aio as grpcclient_aio
import json
import numpy as np
import uuid
import re
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from config_loader import config


# Pydantic Models for validation and cleaning
class CallType(BaseModel):
    human_to_human: bool = False
    ivr: bool = False
    voicemail: bool = False


class Sentiment(BaseModel):
    category: Literal["Positive", "Negative", "Neutral", "Mixed"] = "Neutral"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default = "")
    key_indicators: List[str] = Field(default_factory=list)

    @field_validator('key_indicators', mode='before')
    @classmethod
    def clean_key_indicators(cls, v):
        if not isinstance(v, list):
            return []
        # Remove empty strings, strip whitespace, and remove duplicates
        cleaned = [str(item).strip() for item in v if item and str(item).strip()]
        return list(set(cleaned))

    @field_validator('reasoning', mode='before')
    @classmethod
    def clean_reasoning(cls, v):
        if not isinstance(v, str):
            return ""
        # Remove empty strings, strip whitespace, and remove duplicates
        cleaned = str(v)
        return cleaned
    
    
    @field_validator('category', mode='before')
    @classmethod
    def normalize_category(cls, v):
        if not v:
            return "Neutral"
        v_str = str(v).lower()
        if "very positive" in v_str:
            return "Very Positive"
        if "positive" in v_str:
            return "Positive"
        elif "negative" in v_str:
            return "Negative"
        elif "very negative" in v_str:
            return "Very Negative"
        else:
            return "Neutral"


class CallStatus(BaseModel):
    voicemail_message_request: bool = False
    do_not_disturb: bool = False
    wrong_number: bool = False
    callback_requested: bool = False


class Product(BaseModel):
    name: str = "Unknown Product"
    quantity: Optional[int] = None
    mentioned_at: Optional[str] = None
    
    @field_validator('name', mode='before')
    @classmethod
    def clean_name(cls, v):
        if not v or not str(v).strip():
            return "Unknown Product"
        return str(v).strip()
    
    @field_validator('quantity', mode='before')
    @classmethod
    def clean_quantity(cls, v):
        if v is None or str(v).lower() in ['null', 'none', 'n/a']:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None
    
    @field_validator('mentioned_at', mode='before')
    @classmethod
    def clean_mentioned_at(cls, v):
        if not v or not str(v).strip() or str(v).lower() in ['null', 'none', 'n/a']:
            return None
        return str(v).strip()


class ActionItem(BaseModel):
    description: str = "No description"
    owner: Optional[str] = None
    
    @field_validator('description', mode='before')
    @classmethod
    def clean_description(cls, v):
        if not v or not str(v).strip():
            return "No description"
        return str(v).strip()


class ContactInfo(BaseModel):
    phone: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    
    @field_validator('phone', mode='before')
    @classmethod
    def clean_phone(cls, v):
        if not v or not str(v).strip() or str(v).lower() in ['null', 'none', 'n/a', 'unknown']:
            return None
        return str(v).strip()
    
    @field_validator('email', mode='before')
    @classmethod
    def clean_email(cls, v):
        if not v or not str(v).strip():
            return None
        email_str = str(v).strip().lower()
        if email_str in ['null', 'none', 'n/a', 'unknown'] or '@' not in email_str:
            return None
        return email_str
    
    @field_validator('name', mode='before')
    @classmethod
    def clean_name(cls, v):
        if not v or not str(v).strip() or str(v).lower() in ['null', 'none', 'n/a', 'unknown']:
            return None
        return str(v).strip()


class PersonalInfo(BaseModel):
    type: str
    value: str
    
    @field_validator('type', 'value', mode='before')
    @classmethod
    def clean_strings(cls, v):
        if not v or not str(v).strip():
            return ""
        return str(v).strip()


class Opportunity(BaseModel):
    description: str
    potential_value: Optional[str] = None
    
    @field_validator('description', mode='before')
    @classmethod
    def clean_description(cls, v):
        return str(v).strip() if v else ""


class Risk(BaseModel):
    description: str
    mitigation_strategy: Optional[str] = None
    
    @field_validator('description', mode='before')
    @classmethod
    def clean_description(cls, v):
        return str(v).strip() if v else ""


class BusinessIntelligence(BaseModel):
    opportunities: List[Opportunity] = Field(default_factory=list)
    risks: List[Risk] = Field(default_factory=list)


class ImprovementRecommendation(BaseModel):
    area_for_improvement: str
    recommendation: str
    
    @field_validator('area_for_improvement', 'recommendation', mode='before')
    @classmethod
    def clean_strings(cls, v):
        return str(v).strip() if v else ""


class ExtractedItems(BaseModel):
    products: List[Product] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    contact_info: ContactInfo = Field(default_factory=ContactInfo)
    personal_info: List[PersonalInfo] = Field(default_factory=list)  # Added for compatibility
    
    @field_validator('products', 'action_items', 'personal_info', mode='before')
    @classmethod
    def ensure_list(cls, v):
        if not isinstance(v, list):
            return []
        return v


class FraudDetection(BaseModel):
    # Support both formats
    suspicious_language: bool = False
    high_pressure_tactics: bool = False
    potential_fraud: Optional[bool] = None  # Alternative field name
    risk_level: Literal["Low", "Medium", "High"] = "Low"
    evidence: List[str] = Field(default_factory=list)
    reason: Optional[str] = None  # Alternative field name
    
    @field_validator('risk_level', mode='before')
    @classmethod
    def normalize_risk_level(cls, v):
        if not v:
            return "Low"
        v_str = str(v).lower()
        if "high" in v_str:
            return "High"
        elif "medium" in v_str or "moderate" in v_str:
            return "Medium"
        else:
            return "Low"
    
    @field_validator('evidence', mode='before')
    @classmethod
    def clean_evidence(cls, v):
        if not isinstance(v, list):
            return []
        # Remove empty strings, strip whitespace, and remove duplicates
        cleaned = [str(item).strip() for item in v if item and str(item).strip()]
        return list(set(cleaned))
    
    @model_validator(mode='after')
    def sync_fraud_fields(self):
        """Synchronize potential_fraud with suspicious_language if one is set"""
        if self.potential_fraud is not None and not self.suspicious_language:
            self.suspicious_language = self.potential_fraud
        
        # If reason is provided but evidence is empty, add reason to evidence
        if self.reason and not self.evidence:
            self.evidence = [self.reason]
        
        return self


class AIAnalysis(BaseModel):
    call_type: CallType = Field(default_factory=CallType)
    sentiment: Sentiment = Field(default_factory=Sentiment)
    summary: str = "No summary available"
    call_status: CallStatus = Field(default_factory=CallStatus)
    extracted_items: ExtractedItems = Field(default_factory=ExtractedItems)
    fraud_detection: FraudDetection = Field(default_factory=FraudDetection)
    # Optional fields that may appear in extended outputs
    business_intelligence: Optional[BusinessIntelligence] = None
    improvement_recommendations: List[ImprovementRecommendation] = Field(default_factory=list)
    
    @field_validator('summary', mode='before')
    @classmethod
    def clean_summary(cls, v):
        if not v or not str(v).strip():
            return "No summary available"
        # Remove excessive whitespace
        return " ".join(str(v).split())


class Analysis(BaseModel):
    ai_analysis: AIAnalysis
    
    class Config:
        # Allow extra fields that might be present but not in schema
        extra = 'allow'


class AnalysisResponse(BaseModel):
    request_id: str
    success: bool
    analysis: Analysis


class AsyncAnalysis:
    def __init__(self):
        # Load configuration
        service_config = config.get_service_config('analysis')
        
        self.url = service_config.get('url', 'localhost:3701')
        self.model_name = service_config.get('model_name', 'mistral-nemo')
        self.client = None
        self.system_prompt = """You are an expert AI system specializing in call center analytics with advanced capabilities in:
- Fraud detection and risk assessment
- Sentiment analysis with confidence scoring (5-level scale: Very Positive, Positive, Neutral, Negative, Very Negative)
- Entity and information extraction
- Business intelligence and opportunity identification
- Call quality assessment and improvement recommendations
- Extracting Action Items

Sentiment Classification Guidelines:
- Very Positive: Customer expresses strong satisfaction, gratitude, excitement, or loyalty. Uses emphatic positive language.
- Positive: Customer is satisfied, pleased, or agreeable. Generally cooperative tone.
- Neutral: Customer is matter-of-fact, neither positive nor negative. Transactional interactions.
- Negative: Customer expresses dissatisfaction, frustration, or complaints. Unhappy but manageable.
- Very Negative: Customer is angry, hostile, threatens to leave, or uses strong negative language.

Your responses must be precise, structured JSON that captures both high-level insights and granular details."""
        
    async def initialize(self):
        """Initialize the async Triton client."""
        if self.client is None:
            self.client = grpcclient_aio.InferenceServerClient(url=self.url)
    
    def __preprocess_output(self, output_text):
        """
        Preprocess the LLM output to extract clean JSON.
        Handles cases where JSON is embedded in markdown, mixed with other text, or incomplete.
        """
        try:
            # Remove markdown code blocks first
            output_text = re.sub(r'```json\s*', '', output_text)
            output_text = re.sub(r'```\s*', '', output_text)
            output_text = output_text.strip()
            output_text = re.sub(r'(?<!:)//(?!/)[^\n]*', '', output_text)
            
            # Strategy 1: Find all complete JSON objects containing "ai_analysis"
            # This regex finds balanced braces containing "ai_analysis"
            json_pattern = r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}[^{}]*)*\}'
            
            # Find all potential JSON objects
            potential_jsons = []
            for match in re.finditer(json_pattern, output_text):
                json_str = match.group(0)
                # Check if this JSON contains "ai_analysis"
                if '"ai_analysis"' in json_str:
                    try:
                        parsed = json.loads(json_str)
                        # Verify it has the expected structure
                        if isinstance(parsed, dict) and 'ai_analysis' in parsed:
                            potential_jsons.append((len(json_str), parsed))
                    except json.JSONDecodeError:
                        continue
            # Return the longest valid JSON (most complete)
            if potential_jsons:
                potential_jsons.sort(key=lambda x: x[0], reverse=True)
                return potential_jsons[0][1]
            
            # Strategy 2: Look for the last occurrence of a JSON block with ai_analysis
            # Split by common separators and try each chunk
            chunks = output_text.split('```')
            for chunk in reversed(chunks):
                chunk = chunk.strip()
                if not chunk or not (chunk.startswith('{') or '{' in chunk):
                    continue
                
                # Find the first { and last }
                start_idx = chunk.find('{')
                end_idx = chunk.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = chunk[start_idx:end_idx]
                    if '"ai_analysis"' in json_str:
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and 'ai_analysis' in parsed:
                                return parsed
                        except json.JSONDecodeError:
                            continue
            # Strategy 3: Try to extract from the entire text
            # Look for the rightmost complete JSON that contains ai_analysis
            brace_stack = []
            json_starts = []
            
            for i, char in enumerate(output_text):
                if char == '{':
                    brace_stack.append(i)
                elif char == '}' and brace_stack:
                    start = brace_stack.pop()
                    if not brace_stack:  # Complete JSON object
                        json_str = output_text[start:i+1]
                        if '"ai_analysis"' in json_str:
                            json_starts.append((start, i+1, json_str))
            # Try JSON objects from last to first (most recent/complete)
            for start, end, json_str in reversed(json_starts):
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and 'ai_analysis' in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
            return None
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def __postprocess_with_pydantic(self, raw_analysis_dict, request_id):
        """
        Post-process using Pydantic models for validation and cleaning.
        """
        try:
            # Handle case where ai_analysis might be nested or at root level
            if "ai_analysis" in raw_analysis_dict:
                analysis_data = raw_analysis_dict
            else:
                # Wrap it if it's not already wrapped
                analysis_data = {"ai_analysis": raw_analysis_dict}
            
            # Validate and clean using Pydantic
            analysis = Analysis(**analysis_data)
            
            # Create the response object
            response = AnalysisResponse(
                request_id=request_id,
                success=True,
                analysis=analysis
            )
            
            # Convert to JSON with proper formatting
            return response.model_dump_json(indent=2, exclude_none=False)
            
        except Exception as e:
            print(f"Pydantic validation error: {e}")
            # Return error response
            return json.dumps({
                "request_id": request_id,
                "success": False,
                "error": f"Validation failed: {str(e)}",
                "raw_data": raw_analysis_dict
            }, indent=2)
    
    def __clean_output(self, output_text, request_id):
        """Clean and format the output to ensure consistent structure."""
        try:
            # First, try to preprocess and extract clean JSON
            raw_analysis = self.__preprocess_output(output_text)
            if raw_analysis:
                # Use Pydantic for post-processing
                return self.__postprocess_with_pydantic(raw_analysis, request_id)
            
            # If preprocessing didn't work, provide detailed error
            print(f"Preprocessing failed for request {request_id}")
            print(f"Output text length: {len(output_text)}")
            print(f"First 200 chars: {output_text[:200]}")
            print(f"Last 200 chars: {output_text[-200:]}")
            
            # Try one more fallback: manual JSON extraction
            # Look for ```json blocks specifically
            json_block_pattern = r'```json\s*(\{.*?\})\s*```'
            json_matches = re.finditer(json_block_pattern, output_text, re.DOTALL)
            
            for match in json_matches:
                try:
                    json_str = match.group(1)
                    raw_analysis = json.loads(json_str)
                    if 'ai_analysis' in raw_analysis:
                        return self.__postprocess_with_pydantic(raw_analysis, request_id)
                except json.JSONDecodeError:
                    continue
            
            # Final fallback: try to extract any valid JSON with ai_analysis
            # by finding balanced braces
            depth = 0
            start_pos = -1
            
            for i, char in enumerate(output_text):
                if char == '{':
                    if depth == 0:
                        start_pos = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start_pos != -1:
                        json_str = output_text[start_pos:i+1]
                        if '"ai_analysis"' in json_str:
                            try:
                                raw_analysis = json.loads(json_str)
                                if 'ai_analysis' in raw_analysis:
                                    return self.__postprocess_with_pydantic(raw_analysis, request_id)
                            except json.JSONDecodeError:
                                pass
            
            # If all else fails, return error
            return json.dumps({
                "request_id": request_id,
                "success": False,
                "error": "Could not extract valid JSON from output",
                "debug_info": {
                    "output_length": len(output_text),
                    "contains_ai_analysis": '"ai_analysis"' in output_text,
                    "first_100_chars": output_text[:100],
                    "last_100_chars": output_text[-100:]
                }
            }, indent=2)
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return json.dumps({
                "request_id": request_id,
                "success": False,
                "error": f"Failed to parse JSON: {str(e)}",
                "raw_output": output_text[:500]  # Include first 500 chars for debugging
            }, indent=2)
        except Exception as e:
            print(f"Unexpected error in __clean_output: {e}")
            return json.dumps({
                "request_id": request_id,
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "raw_output": output_text[:500]
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
    
    def __format_prompt(self, user_prompt: str) -> str:
        """Format prompts according to Mistral's instruction format"""
        return f"""[INST] {self.system_prompt}

{user_prompt} [/INST]"""
    
    async def analyze_call_async(self, transcription: str, request_id=None) -> str:
        """
        Analyze call center transcription asynchronously.
        
        Args:
            transcription: RAW transcription text WITHOUT speaker labels
        """
        user_prompt = f"""You are an expert AI system specializing in call center analytics, fraud detection, Action Items and business intelligence extraction.
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
            "category": "Very Positive|Positive|Neutral|Negative|Very Negative",
            "confidence": 0.0-1.0,
            "reasoning": Detailed Reason for the category assigned,
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
                    "description": "what needs to be done and when",
                    "owner": "who is responsible Customer or Agent?"
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
    
        full_prompt = self.__format_prompt(user_prompt)
        return await self.__generate_async(full_prompt, request_id)