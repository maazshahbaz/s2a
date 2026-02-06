import json
import re
from typing import Dict, List, Optional, Literal
import numpy as np
import tritonclient.grpc.aio as grpcclient_aio
from pydantic import BaseModel, Field, field_validator, model_validator
from .config_loader import config


# Pydantic Models for fraud detection validation
class FraudDetection(BaseModel):
    """
    Enhanced fraud detection schema for post-call transcript analysis.
    Detects content-based fraud signals including impersonation, social engineering,
    phishing, and policy violations through transcript analysis.
    """

    # ===== OVERALL FRAUD ASSESSMENT =====
    fraud_detected: bool = False
    risk_level: Literal["Low", "Medium", "High"] = "Low"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in fraud assessment (0-1)")
    fraud_category: Optional[Literal[
        "impersonation",
        "social_engineering",
        "financial_fraud",
        "identity_phishing",
        "policy_violation",
        "none"
    ]] = "none"

    # ===== IMPERSONATION DETECTION =====
    impersonation_detected: bool = False
    impersonation_type: Optional[Literal[
        "authority_figure",  # IRS, police, government official
        "company_representative",  # Bank, IT support, customer service
        "known_contact",  # Friend, family member
        "none"
    ]] = "none"

    # ===== SOCIAL ENGINEERING TACTICS =====
    urgency_tactics: bool = False  # Time pressure, immediate action required
    fear_tactics: bool = False  # Threats, consequences, legal action
    authority_tactics: bool = False  # Claims of authority or official capacity
    scarcity_tactics: bool = False  # Limited time offers, last chance

    # ===== FINANCIAL FRAUD =====
    payment_request_detected: bool = False
    payment_method: Optional[Literal[
        "credit_card",
        "wire_transfer",
        "gift_card",
        "cryptocurrency",
        "cash",
        "bank_transfer",
        "none"
    ]] = "none"

    # ===== IDENTITY PHISHING =====
    identity_verification_request: bool = False
    sensitive_info_requested: List[str] = Field(default_factory=lambda: ["none"])

    # ===== HIGH-PRESSURE TACTICS =====
    high_pressure_tactics: bool = False
    emotional_manipulation: bool = False
    repeated_demands: bool = False

    # ===== POLICY VIOLATIONS =====
    policy_violation_detected: bool = False
    violation_type: Optional[Literal[
        "threats",
        "harassment",
        "inappropriate_language",
        "coercion",
        "discrimination",
        "none"
    ]] = "none"

    # ===== CONVERSATION FLOW ANOMALIES =====
    abrupt_call_ending: bool = False
    scripted_responses_detected: bool = False
    evasive_behavior: bool = False
    inconsistent_information: bool = False

    # ===== EVIDENCE & REASONING =====
    evidence: List[str] = Field(default_factory=list, description="Specific phrases or patterns indicating fraud")
    reasoning: str = Field(default="", description="Detailed explanation of fraud assessment")
    red_flags: List[str] = Field(default_factory=list, description="List of red flags identified")

    # ===== GRACEFUL FALLBACK =====
    data_sufficient: bool = True
    fallback_reason: Optional[str] = None

    # ===== LEGACY FIELDS (for backward compatibility) =====
    suspicious_language: bool = False
    potential_fraud: Optional[bool] = None
    reason: Optional[str] = None

    # ===== VALIDATORS =====

    @field_validator('confidence_score', mode='before')
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence score is between 0 and 1"""
        try:
            score = float(v)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.0

    @field_validator('risk_level', mode='before')
    @classmethod
    def normalize_risk_level(cls, v):
        """Normalize risk level to Low/Medium/High"""
        if not v:
            return "Low"
        v_str = str(v).lower()
        if "high" in v_str or "critical" in v_str or "severe" in v_str:
            return "High"
        elif "medium" in v_str or "moderate" in v_str or "elevated" in v_str:
            return "Medium"
        else:
            return "Low"

    @field_validator('evidence', 'red_flags', mode='before')
    @classmethod
    def clean_string_lists(cls, v):
        """Clean and deduplicate string lists"""
        if not isinstance(v, list):
            return []
        cleaned = [str(item).strip() for item in v if item and str(item).strip()]
        return list(dict.fromkeys(cleaned))  # Preserve order while deduplicating

    @field_validator('sensitive_info_requested', mode='before')
    @classmethod
    def clean_sensitive_info(cls, v):
        """Clean sensitive info list and ensure 'none' if empty"""
        if not isinstance(v, list):
            return ["none"]
        cleaned = [str(item).strip().lower() for item in v if item and str(item).strip()]
        return cleaned if cleaned else ["none"]

    @field_validator('fraud_category', 'impersonation_type', 'payment_method', 'violation_type', mode='before')
    @classmethod
    def normalize_category_fields(cls, v):
        """Normalize category fields to lowercase"""
        if not v or str(v).lower() in ['null', 'none', 'n/a']:
            return "none"
        return str(v).lower().replace(" ", "_")

    @model_validator(mode='after')
    def sync_fields_and_validate(self):
        """
        Synchronize legacy fields, validate fraud detection logic, and implement graceful fallback
        """
        # Sync legacy fields with new fields
        if self.fraud_detected and not self.suspicious_language:
            self.suspicious_language = self.fraud_detected

        if self.potential_fraud is not None and not self.fraud_detected:
            self.fraud_detected = self.potential_fraud

        if self.reasoning and not self.reason:
            self.reason = self.reasoning
        elif self.reason and not self.reasoning:
            self.reasoning = self.reason

        # If any specific fraud indicator is True, fraud_detected should be True
        fraud_indicators = [
            self.impersonation_detected,
            self.urgency_tactics,
            self.fear_tactics,
            self.payment_request_detected,
            self.identity_verification_request,
            self.policy_violation_detected,
        ]

        if any(fraud_indicators) and not self.fraud_detected:
            self.fraud_detected = True

        # Auto-categorize fraud if not set
        if self.fraud_detected and self.fraud_category == "none":
            if self.impersonation_detected:
                self.fraud_category = "impersonation"
            elif self.payment_request_detected:
                self.fraud_category = "financial_fraud"
            elif self.identity_verification_request:
                self.fraud_category = "identity_phishing"
            elif self.policy_violation_detected:
                self.fraud_category = "policy_violation"
            elif any([self.urgency_tactics, self.fear_tactics, self.authority_tactics]):
                self.fraud_category = "social_engineering"

        # Auto-adjust risk level based on fraud category
        if self.fraud_category in ["impersonation", "financial_fraud", "identity_phishing"]:
            if self.risk_level == "Low":
                self.risk_level = "Medium"

        if self.policy_violation_detected and self.violation_type in ["threats", "coercion"]:
            if self.risk_level != "High":
                self.risk_level = "High"

        # Graceful fallback: if confidence is very low, set fallback
        if self.confidence_score < 0.3 and not self.fallback_reason:
            self.fallback_reason = "Low confidence in fraud assessment due to insufficient indicators"
            self.data_sufficient = False

        # If data is insufficient and no strong indicators, reset to safe defaults
        if not self.data_sufficient and not any(fraud_indicators):
            self.fraud_detected = False
            self.risk_level = "Low"
            if not self.fallback_reason:
                self.fallback_reason = "Insufficient data for confident fraud assessment"

        return self


class AsyncFraudDetectionClient:
    """Async gRPC Client for Triton Fraud Detection Server"""

    def __init__(self, url: str = None):
        service_config = config.get_service_config('fraud_detection')

        self.url = url or service_config.get('url', 'localhost:3701')
        self.model_name = service_config.get('model_name', 'mistral-nemo')
        self.client = None
        self.system_prompt = """You are an expert AI fraud detection system specializing in analyzing call center transcripts for fraudulent activity.
You detect content-based fraud signals including impersonation, social engineering, phishing, high-pressure tactics, and policy violations.

Your analysis must be thorough, evidence-based, and structured. You never flag legitimate business conversations as fraud.

CONFIDENCE SCORING:
- High confidence (0.8-1.0): Multiple clear fraud indicators present
- Medium confidence (0.5-0.79): Some fraud indicators but could be legitimate
- Low confidence (0.0-0.49): Insufficient evidence or unclear context

GRACEFUL FALLBACK:
- If transcript is too short (<50 words), set data_sufficient=false
- If unclear or ambiguous, set confidence_score low and explain in fallback_reason
- When in doubt, err on the side of caution with Medium risk rather than High
- Never flag legitimate business conversations as fraud

Your responses must be precise, structured JSON."""

    async def connect(self):
        """Connect to Triton server"""
        self.client = grpcclient_aio.InferenceServerClient(url=self.url)

        if not await self.client.is_server_live():
            raise Exception(f"Triton server at {self.url} is not live")

        if not await self.client.is_server_ready():
            raise Exception(f"Triton server at {self.url} is not ready")

        if not await self.client.is_model_ready(self.model_name):
            raise Exception(f"Model {self.model_name} is not ready")

        print(f"[Fraud Detection] Connected to Triton server at {self.url}")

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

            # Strategy 1: Find JSON objects containing "fraud_detection" or "fraud_detected"
            json_pattern = r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}[^{}]*)*\}'

            potential_jsons = []
            for match in re.finditer(json_pattern, output_text):
                json_str = match.group(0)
                if '"fraud_detection"' in json_str or '"fraud_detected"' in json_str:
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            # If wrapped in "fraud_detection", unwrap
                            if 'fraud_detection' in parsed:
                                potential_jsons.append((len(json_str), parsed['fraud_detection']))
                            elif 'fraud_detected' in parsed:
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
                        if '"fraud_detected"' in json_str or '"fraud_detection"' in json_str:
                            json_starts.append((start, i+1, json_str))

            for start, end, json_str in reversed(json_starts):
                try:
                    parsed = json.loads(json_str)
                    if 'fraud_detection' in parsed:
                        return parsed['fraud_detection']
                    if 'fraud_detected' in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue

            return None

        except Exception as e:
            print(f"[Fraud Detection] Preprocessing error: {e}")
            return None

    def __postprocess_with_pydantic(self, raw_dict: Dict) -> Dict:
        """Post-process using Pydantic FraudDetection model for validation and cleaning."""
        try:
            fraud_result = FraudDetection(**raw_dict)
            return json.loads(fraud_result.model_dump_json())

        except Exception as e:
            print(f"[Fraud Detection] Pydantic validation error: {e}")
            return self.__get_default_response()

    def __get_default_response(self) -> Dict:
        """Return safe default fraud detection response."""
        return json.loads(FraudDetection().model_dump_json())

    def __clean_output(self, output_text: str) -> Dict:
        """Clean and format the output to ensure consistent structure."""
        try:
            raw_fraud = self.__preprocess_output(output_text)
            if raw_fraud:
                return self.__postprocess_with_pydantic(raw_fraud)

            print(f"[Fraud Detection] Preprocessing failed")
            print(f"[Fraud Detection] Output text length: {len(output_text)}")
            print(f"[Fraud Detection] First 200 chars: {output_text[:200]}")

            return self.__get_default_response()

        except Exception as e:
            print(f"[Fraud Detection] Unexpected error in __clean_output: {e}")
            return self.__get_default_response()

    async def detect_fraud(self, transcript: str, request_id: str) -> Dict:
        """
        Analyze a call transcript for fraud indicators.

        Args:
            transcript: The call transcript text
            request_id: Unique request identifier

        Returns:
            Dict with fraud detection results
        """
        if not self.client:
            await self.connect()

        user_prompt = f"""Analyze the following call transcription for fraud indicators.

Call Transcription:
{transcript}

=== FRAUD DETECTION GUIDELINES ===

Carefully analyze the transcript for fraud indicators. Look for these specific patterns:

1. IMPERSONATION:
   - Authority Figures: Claims to be IRS, police, government official, FBI, DEA, court
   - Company Representatives: Fake bank, IT support, tech support, utility company, insurance
   - Known Contacts: Pretending to be family, friend, colleague
   Examples: "This is the IRS", "I'm from Microsoft", "calling from your bank's fraud department"

2. SOCIAL ENGINEERING TACTICS:
   - Urgency: "Act immediately", "within 24 hours", "deadline today", "right now"
   - Fear: "Legal action", "arrest warrant", "account frozen", "service disconnection"
   - Authority: "Final notice", "mandatory", "required by law", "official investigation"
   - Scarcity: "Limited time", "one-time offer", "last chance", "expires soon"

3. FINANCIAL FRAUD:
   - Payment Requests: Credit card numbers, wire transfers, gift cards, cryptocurrency
   - Immediate Payment: "Pay now", "send payment", "transfer funds", "purchase gift cards"
   - Unusual Methods: iTunes/Google Play cards, Bitcoin, Western Union, untraceable methods
   Examples: "Pay with gift cards", "wire the money", "give me your card number"

4. IDENTITY PHISHING:
   - Sensitive Information: SSN, passwords, PINs, verification codes, security questions
   - Account Credentials: Username, password, account numbers, routing numbers
   - Personal Data: Date of birth, mother's maiden name, full SSN
   Examples: "Verify your identity with your SSN", "what's your password", "provide verification code"

5. HIGH-PRESSURE TACTICS:
   - Emotional Manipulation: Creating panic, urgency, fear, guilt
   - Repeated Demands: Asking multiple times for same information
   - Preventing Verification: "Don't hang up", "don't tell anyone", "do it now"

6. POLICY VIOLATIONS:
   - Threats: Physical harm, legal consequences, reputation damage
   - Harassment: Aggressive language, insults, intimidation
   - Coercion: Forcing actions against will, preventing informed decisions

7. CONVERSATION ANOMALIES:
   - Abrupt Ending: Call ends suddenly after sensitive request
   - Scripted Responses: Robotic, repetitive, ignoring questions
   - Evasiveness: Avoiding direct answers, deflecting questions
   - Inconsistencies: Conflicting information, changing story

Provide your analysis following this EXACT JSON structure. Return ONLY valid JSON with no additional text:
{{
    "fraud_detection": {{
        "fraud_detected": true/false,
        "risk_level": "Low|Medium|High",
        "confidence_score": 0.0-1.0,
        "fraud_category": "impersonation|social_engineering|financial_fraud|identity_phishing|policy_violation|none",

        "impersonation_detected": true/false,
        "impersonation_type": "authority_figure|company_representative|known_contact|none",

        "urgency_tactics": true/false,
        "fear_tactics": true/false,
        "authority_tactics": true/false,
        "scarcity_tactics": true/false,

        "payment_request_detected": true/false,
        "payment_method": "credit_card|wire_transfer|gift_card|cryptocurrency|cash|bank_transfer|none",

        "identity_verification_request": true/false,
        "sensitive_info_requested": ["ssn", "password", "pin", "verification_code", "security_question", "account_number", "credit_card", "date_of_birth"] or ["none"],

        "high_pressure_tactics": true/false,
        "emotional_manipulation": true/false,
        "repeated_demands": true/false,

        "policy_violation_detected": true/false,
        "violation_type": "threats|harassment|inappropriate_language|coercion|discrimination|none",

        "abrupt_call_ending": true/false,
        "scripted_responses_detected": true/false,
        "evasive_behavior": true/false,
        "inconsistent_information": true/false,

        "evidence": ["specific phrases or patterns indicating fraud"],
        "reasoning": "detailed explanation of fraud assessment",
        "red_flags": ["list of specific red flags identified"],

        "data_sufficient": true/false,
        "fallback_reason": "explanation if confidence is low or data insufficient"
    }}
}}

IMPORTANT:
- Return ONLY valid JSON with no additional text
- Use null for missing values, never omit fields
- Ensure all boolean values are lowercase (true/false)
- Keep arrays empty [] if no items found
- Be specific and detailed in evidence and reasoning"""

        # Format with Mistral instruction template
        prompt = self.__format_prompt(user_prompt)

        print(f"[Fraud Detection] Sending fraud detection request request_id: {request_id}")
        print(f"[Fraud Detection] Prompt length (chars): {len(prompt)}")

        # Create input tensor
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

        # Extract generated text
        raw_output = response.as_numpy("generated_text")
        print(f"[Fraud Detection] Raw output shape: {raw_output.shape}")
        print(f"[Fraud Detection] Raw output dtype: {raw_output.dtype}")

        # Handle different output shapes
        if raw_output.ndim == 2:
            output_text = raw_output[0][0]
        else:
            output_text = raw_output[0]

        if isinstance(output_text, bytes):
            output_text = output_text.decode('utf-8')

        print(f"[Fraud Detection] Received fraud detection response for request_id: {request_id}")
        print(f"[Fraud Detection] Output text length: {len(output_text)}")

        if len(output_text) > 0:
            print(f"[Fraud Detection] First 500 chars of response: {output_text[:500]}")

        # Parse and validate response
        result = self.__clean_output(output_text)

        return {"fraud_detection": result}

    async def close(self):
        """Close client connection"""
        if self.client:
            await self.client.close()
            print("[Fraud Detection] Connection closed")

    def __format_prompt(self, user_prompt: str) -> str:
        """Format prompts according to Mistral's instruction format"""
        return f"""[INST] {self.system_prompt}

{user_prompt} [/INST]"""
