"""
LLM-Based Speaker Label Corrector using Mistral-Nemo via Triton.

Directly assigns Agent/Customer labels instead of speaker_0/speaker_1.
Single optimized class for production use.
"""

import tritonclient.grpc.aio as grpcclient_aio
import numpy as np
import json
import uuid
import re
from typing import List, Dict, Tuple, Optional
import time
from .config_loader import config


class LLMSpeakerCorrector:
    """
    Uses Mistral-Nemo LLM to analyze conversations and assign Agent/Customer labels.
    
    This replaces generic speaker_0/speaker_1 with meaningful Agent/Customer labels
    by understanding the conversation context and participant roles.
    
    Usage:
        corrector = LLMSpeakerCorrector()
        corrected_transcript, corrected_words = await corrector.assign_agent_customer_labels(
            labeled_transcription,
            aligned_words
        )
    """
    
    def __init__(self, triton_url: str = None, model_name: str = None):
        """
        Initialize LLM corrector.
        
        Args:
            triton_url: Triton server URL (default: from config)
            model_name: Model name in Triton (default: from config)
        """
        # Load configuration
        service_config = config.get_service_config('speaker_correction')
        
        self.triton_url = triton_url or service_config.get('url', 'localhost:3701')
        self.model_name = model_name or service_config.get('model_name', 'mistral-nemo')
        self.client = grpcclient_aio.InferenceServerClient(url=self.triton_url)
        print(f"[LLM Corrector] Initialized with Triton at {self.triton_url}")
        
        self.system_prompt = """You are an expert at analyzing customer service call transcripts. Your task is to label each speaker as either AGENT or CUSTOMER.

Guidelines:

AGENT usually:
- Greets first (e.g., "Good morning", "Thank you for calling")
- States their name or company
- Asks "How can I help you?" or similar
- Gives information, solutions, or assistance
- Uses professional, formal language
- Asks questions to understand the issue

CUSTOMER usually:
- Responds to greetings
- Explains their problem or question
- Provides personal information (name, order number, account details)
- Asks about products or services
- May show frustration or urgency
- Uses casual or conversational language

Read the transcript and assign the correct role to each speaker."""
    
    async def initialize(self):
        """Initialize Triton client."""
        if self.client is None:
            self.client = grpcclient_aio.InferenceServerClient(url=self.triton_url)
    
    def _format_prompt(self, user_prompt: str) -> str:
        """Format prompts according to Mistral's instruction format."""
        return f"""[INST] {self.system_prompt}

{user_prompt} [/INST]"""
    
    async def _generate_async(self, prompt: str, request_id: str = None) -> str:
        """Generate text using Mistral-Nemo via Triton."""
        # await self.initialize()
        
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
        
        return output_text
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response (handles markdown code blocks and various formats)."""
        # Try to extract from ```json blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to extract from ``` blocks without json tag
        json_match = re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    async def analyze_speaker_roles(
        self,
        request_id: str,
        labeled_transcription: str,
        sample_size: int = 20
    ) -> Dict:
        """
        Analyze the conversation and determine which speaker is Agent vs Customer.
        
        Args:
            labeled_transcription: Transcription with [speaker_0], [speaker_1] labels
            sample_size: Number of lines to analyze (default: 20, enough for most cases)
            
        Returns:
            {
                'speaker_0_role': 'Agent' or 'Customer',
                'speaker_1_role': 'Agent' or 'Customer',
                'confidence': float (0.0-1.0),
                'reasoning': str
            }
        """
        t1 = time.time()

        # Take first N lines for analysis
        lines = labeled_transcription.split('\n')
        sample_transcript = '\n'.join(lines)
        

        user_prompt = f"""Analyze this customer service call transcript and identify which speaker is the Agent and which is the Customer.

TRANSCRIPT:
{sample_transcript}

Respond with ONLY a JSON object (no other text):
{{
    "speaker_0_role": "Agent" or "Customer",
    "speaker_1_role": "Agent" or "Customer",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation focusing on key phrases that indicate each role"
}}

Important: One speaker must be "Agent" and the other must be "Customer"."""

        full_prompt = self._format_prompt(user_prompt)
        
        print("[LLM Corrector] Analyzing speaker roles with Mistral-Nemo...")
        response = await self._generate_async(full_prompt, request_id)
        
        # Extract JSON from response
        result = self._extract_json_from_response(response)
        
        if result is None:
            print("[LLM Corrector] Warning: Could not parse LLM response")
            print(f"[LLM Corrector] Response preview: {response[:300]}...")
            
            # Fallback: Use simple heuristic (first speaker is usually agent)
            print("[LLM Corrector] Using fallback: assuming speaker_0 is Agent")
            return {
                'speaker_0_role': 'Agent',
                'speaker_1_role': 'Customer',
                'confidence': 0.5,
                'reasoning': 'Fallback: Could not parse LLM response, assuming standard pattern'
            }
        
        # Validate response
        if result.get('speaker_0_role') not in ['Agent', 'Customer']:
            result['speaker_0_role'] = 'Agent'
        if result.get('speaker_1_role') not in ['Agent', 'Customer']:
            result['speaker_1_role'] = 'Customer'
        
        # Ensure one is Agent and one is Customer
        if result['speaker_0_role'] == result['speaker_1_role']:
            print("[LLM Corrector] Warning: Both speakers assigned same role, correcting...")
            if result['speaker_0_role'] == 'Agent':
                result['speaker_1_role'] = 'Customer'
            else:
                result['speaker_0_role'] = 'Agent'
        
        print(f"[LLM Corrector] Analysis complete:")
        print(f"  speaker_0 → {result['speaker_0_role']}")
        print(f"  speaker_1 → {result['speaker_1_role']}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')}...")
        print(f"Time taken: {time.time() - t1}")
        return result
    
    async def assign_agent_customer_labels(
        self,
        labeled_transcription: str,
        request_id: str,
        aligned_words: Optional[List[Dict]] = None,
        sample_size: int = 20
    ) -> Tuple[str, Optional[List[Dict]]]:
        """
        Analyze conversation and replace speaker_0/speaker_1 with Agent/Customer labels.
        
        Args:
            labeled_transcription: Transcription with [speaker_0], [speaker_1] labels
            aligned_words: Optional list of words with speaker labels
            sample_size: Number of lines to analyze for role detection
            
        Returns:
            (corrected_transcription, corrected_aligned_words)
            where labels are now [Agent] and [Customer] instead of [speaker_0] and [speaker_1]
        """
        
        # Analyze speaker roles
        analysis = await self.analyze_speaker_roles(request_id, labeled_transcription, sample_size,)
        
        # Create label mapping
        label_map = {
            'speaker_0': analysis['speaker_0_role'],
            'speaker_1': analysis['speaker_1_role']
        }
        
        print(f"[LLM Corrector] Applying labels: speaker_0→{label_map['speaker_0']}, speaker_1→{label_map['speaker_1']}")
        
        # Replace labels in transcription
        corrected_transcription = labeled_transcription
        corrected_transcription = corrected_transcription.replace('[speaker_0]', f'[{label_map["speaker_0"]}]')
        corrected_transcription = corrected_transcription.replace('[speaker_1]', f'[{label_map["speaker_1"]}]')
        
        # Replace labels in aligned words if provided
        corrected_words = None
        if aligned_words:
            corrected_words = []
            for word in aligned_words:
                new_word = word.copy()
                if word['speaker'] == 'speaker_0':
                    new_word['speaker'] = label_map['speaker_0']
                elif word['speaker'] == 'speaker_1':
                    new_word['speaker'] = label_map['speaker_1']
                corrected_words.append(new_word)
        
        return corrected_transcription, corrected_words
    
    async def close(self):
        """Close Triton client connection."""
        if self.client:
            await self.client.close()
            self.client = None