import json
import re
from typing import Dict, List, Optional
import numpy as np
import tritonclient.grpc.aio as grpcclient_aio
from pydantic import BaseModel, Field, field_validator
from .config_loader import config


# Pydantic Models for validation
class ScoringMetrics(BaseModel):
    greeting: int = Field(default=0, ge=0, le=10)
    clarity: int = Field(default=0, ge=0, le=15)
    listening_balance: int = Field(default=0, ge=0, le=10)
    response_relevance: int = Field(default=0, ge=0, le=20)
    tone_professionalism: int = Field(default=0, ge=0, le=15)
    call_flow: int = Field(default=0, ge=0, le=10)
    call_closure: int = Field(default=0, ge=0, le=10)
    compliance: int = Field(default=0, ge=0, le=10)

    @field_validator('*', mode='before')
    @classmethod
    def coerce_to_int(cls, v):
        if v is None:
            return 0
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0


class AgentScoring(BaseModel):
    overall_score: int = Field(default=0, ge=0, le=100)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metrics: ScoringMetrics = Field(default_factory=ScoringMetrics)
    flags: List[str] = Field(default_factory=list)

    @field_validator('overall_score', mode='before')
    @classmethod
    def coerce_overall_score(cls, v):
        if v is None:
            return 0
        try:
            return max(0, min(100, int(v)))
        except (ValueError, TypeError):
            return 0

    @field_validator('confidence_score', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        if v is None:
            return 0.0
        try:
            return max(0.0, min(1.0, float(v)))
        except (ValueError, TypeError):
            return 0.0

    @field_validator('flags', mode='before')
    @classmethod
    def clean_flags(cls, v):
        if not isinstance(v, list):
            return []
        cleaned = [str(item).strip() for item in v if item and str(item).strip()]
        return list(set(cleaned))


class ScoringResponse(BaseModel):
    agent_scoring: AgentScoring


class AsyncCSRScoringClient:
    """Async gRPC Client for Triton CSR Scoring Server"""
    
    def __init__(self, url: str = None):
        service_config = config.get_service_config('csr_scoring')
        
        self.url = url or service_config.get('url', 'localhost:8001')
        self.model_name = service_config.get('model_name', 'mistral-nemo')
        self.client = None
        self.system_prompt = """You are an expert Call Quality Analyst specializing in evaluating customer service representative (CSR) performance. 
You analyze call transcripts and provide detailed scoring based on predefined rubrics.
Your responses must be precise, structured JSON that captures the agent's performance metrics."""
    
    async def connect(self):
        """Connect to Triton server"""
        self.client = grpcclient_aio.InferenceServerClient(url=self.url)
        
        if not await self.client.is_server_live():
            raise Exception(f"Triton server at {self.url} is not live")
        
        if not await self.client.is_server_ready():
            raise Exception(f"Triton server at {self.url} is not ready")
        
        if not await self.client.is_model_ready(self.model_name):
            raise Exception(f"Model {self.model_name} is not ready")
        
        print(f"[Agent Scoring] Connected to Triton server at {self.url}")

    def __preprocess_output(self, output_text: str) -> Optional[Dict]:
        """
        Preprocess the LLM output to extract clean JSON.
        Handles cases where JSON is embedded in markdown or mixed with other text.
        """
        try:
            # Remove markdown code blocks
            output_text = re.sub(r'```json\s*', '', output_text)
            output_text = re.sub(r'```\s*', '', output_text)
            output_text = output_text.strip()
            output_text = re.sub(r'(?<!:)//(?!/)[^\n]*', '', output_text)
            
            # Strategy 1: Find JSON objects containing "agent_scoring"
            json_pattern = r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}[^{}]*)*\}'
            
            potential_jsons = []
            for match in re.finditer(json_pattern, output_text):
                json_str = match.group(0)
                if '"agent_scoring"' in json_str:
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and 'agent_scoring' in parsed:
                            potential_jsons.append((len(json_str), parsed))
                    except json.JSONDecodeError:
                        continue
            
            if potential_jsons:
                potential_jsons.sort(key=lambda x: x[0], reverse=True)
                return potential_jsons[0][1]
            
            # Strategy 2: Try balanced brace extraction
            brace_stack = []
            json_starts = []
            
            for i, char in enumerate(output_text):
                if char == '{':
                    brace_stack.append(i)
                elif char == '}' and brace_stack:
                    start = brace_stack.pop()
                    if not brace_stack:
                        json_str = output_text[start:i+1]
                        if '"agent_scoring"' in json_str or '"overall_score"' in json_str:
                            json_starts.append((start, i+1, json_str))
            
            for start, end, json_str in reversed(json_starts):
                try:
                    parsed = json.loads(json_str)
                    # Wrap if needed
                    if 'agent_scoring' not in parsed and 'overall_score' in parsed:
                        parsed = {'agent_scoring': parsed}
                    if 'agent_scoring' in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
            
            return None
            
        except Exception as e:
            print(f"[Agent Scoring] Preprocessing error: {e}")
            return None

    def __postprocess_with_pydantic(self, raw_dict: Dict) -> Dict:
        """Post-process using Pydantic models for validation and cleaning."""
        try:
            if "agent_scoring" in raw_dict:
                scoring_data = raw_dict
            else:
                scoring_data = {"agent_scoring": raw_dict}
            
            response = ScoringResponse(**scoring_data)
            return json.loads(response.model_dump_json())
            
        except Exception as e:
            print(f"[Agent Scoring] Pydantic validation error: {e}")
            return self.__get_error_response(f"Validation failed: {str(e)}")

    def __get_error_response(self, error_msg: str = "parse_error") -> Dict:
        """Return default error response structure."""
        return {
            "agent_scoring": {
                "overall_score": -1,
                "confidence_score": 0.0,
                "metrics": {
                    "greeting": -1,
                    "clarity": -1,
                    "listening_balance": -1,
                    "response_relevance": -1,
                    "tone_professionalism": -1,
                    "call_flow": -1,
                    "call_closure": -1,
                    "compliance": -1
                },
                "flags": ["parse_error"]
            }
        }

    def __clean_output(self, output_text: str) -> Dict:
        """Clean and format the output to ensure consistent structure."""
        try:
            raw_scoring = self.__preprocess_output(output_text)
            if raw_scoring:
                return self.__postprocess_with_pydantic(raw_scoring)
            
            print(f"[Agent Scoring] Preprocessing failed")
            print(f"[Agent Scoring] Output text length: {len(output_text)}")
            print(f"[Agent Scoring] First 200 chars: {output_text[:200]}")
            
            return self.__get_error_response()
            
        except Exception as e:
            print(f"[Agent Scoring] Unexpected error in __clean_output: {e}")
            return self.__get_error_response()

    async def score_transcript(self, transcript: str, request_id: str) -> Dict:
        """
        Score a CSR agent's performance based on call transcript.
        
        Args:
            transcript: The call transcript text
            request_id: Unique request identifier
            
        Returns:
            Dict with scoring results
        """
        if not self.client:
            await self.connect()
        
        user_prompt = f"""Analyze the following customer service call transcript and score the CSR agent's performance.

## Scoring Rubric:

1. **Call Opening & Greeting (10 points max)**: Polite greeting, self-introduction, clear opening
2. **Communication Clarity (15 points max)**: Clear language, minimal filler words, understandable explanations
3. **Listening vs Talking Balance (10 points max)**: Agent allows customer to speak, minimal interruptions
4. **Response Relevance (20 points max)**: Responses align with customer intent, no irrelevant or generic replies
5. **Tone & Professionalism (15 points max)**: Calm, respectful tone, no rude or aggressive language
6. **Call Flow & Structure (10 points max)**: Logical progression, smooth transitions
7. **Call Closure (10 points max)**: Proper closing, summary or next steps, thank-you statement
8. **Compliance & Red Flags (10 points max)**: No policy violations, no prohibited phrases

## Transcript:
{transcript}

## Instructions:
Analyze the transcript and provide scores for each metric. Return ONLY a valid JSON object with no additional text, following this exact structure:

{{
  "agent_scoring": {{
    "overall_score": <sum of all metric scores, 0-100>,
    "confidence_score": <your confidence in the scoring, 0.0-1.0>,
    "metrics": {{
      "greeting": <0-10>,
      "clarity": <0-15>,
      "listening_balance": <0-10>,
      "response_relevance": <0-20>,
      "tone_professionalism": <0-15>,
      "call_flow": <0-10>,
      "call_closure": <0-10>,
      "compliance": <0-10>
    }},
    "flags": [<list of any issues detected, e.g., "minor_interruption", "missing_greeting", "policy_violation">]
  }}
}}

IMPORTANT:
- Return ONLY valid JSON with no additional text
- Ensure all numeric scores are within their specified ranges
- Keep flags array empty [] if no issues found"""

        # Format with Mistral instruction template
        prompt = self.__format_prompt(user_prompt)
        
        print(f"[Agent Scoring] Sending scoring request request_id: {request_id}")
        print(f"[Agent Scoring] Prompt length (chars): {len(prompt)}")
        
        # Create input tensor - matching analysis_client.py format
        input_data = np.array([[prompt]], dtype=object)
        inputs = [grpcclient_aio.InferInput("prompt", [1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(input_data)
        
        # Create output
        outputs = [grpcclient_aio.InferRequestedOutput("generated_text")]
        
        # Send async request
        response = await self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id
        )
        
        # Extract generated text - handle both array shapes
        raw_output = response.as_numpy("generated_text")
        print(f"[Agent Scoring] Raw output shape: {raw_output.shape}")
        print(f"[Agent Scoring] Raw output dtype: {raw_output.dtype}")
        
        # Handle different output shapes
        if raw_output.ndim == 2:
            output_text = raw_output[0][0]
        else:
            output_text = raw_output[0]
        
        if isinstance(output_text, bytes):
            output_text = output_text.decode('utf-8')
        
        print(f"[Agent Scoring] Received scoring response for request_id: {request_id}")
        print(f"[Agent Scoring] Output text length: {len(output_text)}")
        
        if len(output_text) > 0:
            print(f"[Agent Scoring] First 500 chars of response: {output_text[:500]}")
        
        # Parse and validate response
        result = self.__clean_output(output_text)
        
        return result
    
    async def close(self):
        """Close client connection"""
        if self.client:
            await self.client.close()
            print("[Agent Scoring] Connection closed")

    def __format_prompt(self, user_prompt: str) -> str:
        """Format prompts according to Mistral's instruction format"""
        return f"""[INST] {self.system_prompt}

{user_prompt} [/INST]"""