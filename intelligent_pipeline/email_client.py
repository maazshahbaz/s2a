import tritonclient.grpc.aio as grpcclient_aio
import json
import numpy as np
import uuid
import re
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from .config_loader import config


# ============================================================
# Pydantic Models — Simplified Output
# ============================================================


class FollowUpEmail(BaseModel):
    """The generated follow-up email."""
    subject: str = ""
    greeting: str = ""
    body: str = ""
    call_to_action: str = ""
    closing: str = ""
    full_email: str = ""
    tone: Literal[
        "professional",
        "friendly",
        "urgent",
        "consultative",
        "empathetic",
        "assertive"
    ] = "professional"
    email_type: Literal[
        "demo_follow_up",
        "pricing_follow_up",
        "proposal_follow_up",
        "meeting_recap",
        "objection_handling",
        "next_steps_confirmation",
        "thank_you",
        "re_engagement",
        "contract_follow_up",
        "general_follow_up",
        "problem_solving",
        "information_follow_up"
    ] = "general_follow_up"

    @field_validator(
        'subject', 'greeting', 'body', 'call_to_action', 'closing', 'full_email',
        mode='before'
    )
    @classmethod
    def clean_str(cls, v):
        if not v:
            return ""
        return str(v).strip()

    @field_validator('tone', mode='before')
    @classmethod
    def normalize_tone(cls, v):
        if not v:
            return "professional"
        v_str = str(v).strip().lower()
        valid = {"professional", "friendly", "urgent", "consultative", "empathetic", "assertive"}
        return v_str if v_str in valid else "professional"

    @field_validator('email_type', mode='before')
    @classmethod
    def normalize_email_type(cls, v):
        if not v:
            return "general_follow_up"
        return str(v).strip().lower().replace(" ", "_")


class FollowUpEmailOutput(BaseModel):
    """Top-level output — only the fields the caller needs."""
    follow_up_email: FollowUpEmail = Field(default_factory=FollowUpEmail)
    data_sufficient: bool = True

    @model_validator(mode='after')
    def build_full_email_if_missing(self):
        """Assemble full_email from parts if the LLM didn't provide it."""
        email = self.follow_up_email
        if not email.full_email and (email.greeting or email.body):
            parts = [
                p for p in [email.greeting, email.body, email.call_to_action, email.closing]
                if p
            ]
            email.full_email = "\n\n".join(parts)
        return self

    @model_validator(mode='after')
    def check_data_sufficiency(self):
        """If the email body is empty the data was clearly not sufficient."""
        if not self.follow_up_email.body and self.data_sufficient:
            self.data_sufficient = False
        return self


class FollowUpEmailResponse(BaseModel):
    """API-level response wrapper."""
    request_id: str
    success: bool
    output: FollowUpEmailOutput


# ============================================================
# Async Client for Follow-Up Email Generation
# ============================================================


class AsyncFollowUpEmailClient:
    """
    Generates action-driven follow-up emails from call transcriptions.

    Uses the same Triton Inference Server / Mistral-Nemo pattern as
    AsyncAnalysis and AsyncCSRScoringClient.
    """

    def __init__(self):
        service_config = config.get_service_config('followup_email')

        self.url = service_config.get('url', 'localhost:3701')
        self.model_name = service_config.get('model_name', 'mistral-nemo')
        self.client = None

        self.system_prompt = (
            "You are an elite sales enablement AI that converts call transcripts "
            "into highly effective follow-up emails. You combine deep conversation "
            "analysis with proven sales methodologies (MEDDIC, BANT, Challenger Sale, "
            "SPIN Selling) to craft emails that drive the next action.\n\n"
            "Your emails are:\n"
            "- Action-oriented with a single, clear CTA\n"
            "- Personalized using specifics from the conversation\n"
            "- Concise yet comprehensive\n"
            "- Professionally warm without being generic\n"
            "- Strategically structured to overcome identified objections\n\n"
            "You ALWAYS respond with valid JSON only."
        )

    async def initialize(self):
        if self.client is None:
            self.client = grpcclient_aio.InferenceServerClient(url=self.url)

    # ------------------------------------------------------------------
    # JSON extraction (mirrors analysis_client pattern)
    # ------------------------------------------------------------------

    def _preprocess_output(self, output_text: str) -> Optional[dict]:
        """Extract clean JSON from potentially messy LLM output."""
        try:
            output_text = re.sub(r'```json\s*', '', output_text)
            output_text = re.sub(r'```\s*', '', output_text)
            output_text = output_text.strip()
            output_text = re.sub(r'(?<!:)//(?!/)[^\n]*', '', output_text)

            # Strategy 1: regex for balanced JSON containing our key
            json_pattern = (
                r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}[^{}]*)*\}'
            )
            candidates = []
            for match in re.finditer(json_pattern, output_text):
                json_str = match.group(0)
                if '"follow_up_email"' in json_str:
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and 'follow_up_email' in parsed:
                            candidates.append((len(json_str), parsed))
                    except json.JSONDecodeError:
                        continue
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                return candidates[0][1]

            # Strategy 2: brace-matching walk
            brace_stack: list[int] = []
            objects: list[tuple[int, int, str]] = []
            for i, ch in enumerate(output_text):
                if ch == '{':
                    brace_stack.append(i)
                elif ch == '}' and brace_stack:
                    start = brace_stack.pop()
                    if not brace_stack:
                        segment = output_text[start:i + 1]
                        if '"follow_up_email"' in segment:
                            objects.append((start, i + 1, segment))

            for _, _, segment in reversed(objects):
                try:
                    parsed = json.loads(segment)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

            return None
        except Exception as e:
            print(f"[FollowUpEmail] Preprocessing error: {e}")
            return None

    def _postprocess_with_pydantic(self, raw: dict, request_id: str) -> str:
        """Validate with Pydantic, strip to only the required fields, return JSON string."""
        try:
            # The LLM prompt asks for many analysis fields internally but
            # we only keep follow_up_email + data_sufficient in the output.
            trimmed = {
                "follow_up_email": raw.get("follow_up_email", {}),
                "data_sufficient": raw.get("data_sufficient", True),
            }

            output = FollowUpEmailOutput(**trimmed)
            response = FollowUpEmailResponse(
                request_id=request_id,
                success=True,
                output=output,
            )
            return response.model_dump_json(indent=2, exclude_none=False)
        except Exception as e:
            print(f"[FollowUpEmail] Pydantic validation error: {e}")
            return json.dumps({
                "request_id": request_id,
                "success": False,
                "error": f"Validation failed: {e}",
                "raw_data": raw,
            }, indent=2)

    def _clean_output(self, output_text: str, request_id: str) -> str:
        raw = self._preprocess_output(output_text)
        if raw:
            return self._postprocess_with_pydantic(raw, request_id)

        print(f"[FollowUpEmail] Preprocessing failed for request {request_id}")
        print(f"[FollowUpEmail] First 300 chars: {output_text[:300]}")
        return json.dumps({
            "request_id": request_id,
            "success": False,
            "error": "Could not extract valid JSON from LLM output",
            "debug_info": {
                "output_length": len(output_text),
                "contains_follow_up_email": '"follow_up_email"' in output_text,
                "first_100_chars": output_text[:100],
                "last_100_chars": output_text[-100:],
            }
        }, indent=2)

    # ------------------------------------------------------------------
    # Triton inference
    # ------------------------------------------------------------------

    async def _generate_async(self, prompt: str, request_id: Optional[str] = None) -> str:
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
            request_id=request_id,
        )
        output_text = response.as_numpy("generated_text")[0].decode('utf-8')
        if isinstance(output_text, bytes):
            output_text = output_text.decode('utf-8')

        print(output_text)

        return self._clean_output(output_text, request_id)

    def _format_prompt(self, user_prompt: str) -> str:
        return f"[INST] {self.system_prompt}\n\n{user_prompt} [/INST]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_follow_up_email(
        self,
        transcript: str,
        analysis_context: Optional[dict] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Generate a follow-up email from a call transcript.

        Args:
            transcript: Speaker-labeled transcript text.
            analysis_context: Optional pre-computed analysis dict from
                              AsyncAnalysis (sentiment, summary, etc.) to
                              give the model richer context.
            request_id: Unique identifier for this request.

        Returns:
            JSON string conforming to FollowUpEmailResponse.
        """

        # Build optional context block
        context_block = ""
        if analysis_context:
            try:
                ai = analysis_context.get("analysis", {}).get("ai_analysis", {})
                parts = []
                if ai.get("summary"):
                    parts.append(f"Call Summary: {ai['summary']}")
                sentiment = ai.get("sentiment", {})
                if sentiment.get("category"):
                    parts.append(
                        f"Customer Sentiment: {sentiment['category']} "
                        f"(confidence {sentiment.get('confidence', 'N/A')})"
                    )
                products = (
                    ai.get("extracted_items", {}).get("products", [])
                )
                if products:
                    names = [p.get("name", "") for p in products if p.get("name")]
                    if names:
                        parts.append(f"Products Discussed: {', '.join(names)}")
                actions = (
                    ai.get("extracted_items", {}).get("action_items", [])
                )
                if actions:
                    descs = [a.get("description", "") for a in actions if a.get("description")]
                    if descs:
                        parts.append(f"Action Items: {'; '.join(descs)}")
                contact = ai.get("extracted_items", {}).get("contact_info", {})
                if contact.get("name"):
                    parts.append(f"Customer Name: {contact['name']}")
                if contact.get("email"):
                    parts.append(f"Customer Email: {contact['email']}")
                if parts:
                    context_block = (
                        "\n=== PRE-COMPUTED CALL ANALYSIS (use to enrich your response) ===\n"
                        + "\n".join(parts)
                        + "\n=== END PRE-COMPUTED ANALYSIS ===\n"
                    )
            except Exception:
                pass  # Gracefully skip if analysis_context is malformed

        user_prompt = f"""Analyze the following sales call transcript and generate a targeted follow-up email.

{context_block}

=== CALL TRANSCRIPT ===
{transcript}
=== END TRANSCRIPT ===

=== INSTRUCTIONS ===

Perform TWO tasks and return them as a single JSON object:

TASK 1 — CONVERSATION ANALYSIS
Thoroughly analyze the transcript for:

1. INTENT DETECTION
   - Primary buyer intent: purchase, demo_request, pricing_inquiry, information_gathering,
     complaint_resolution, renewal, upsell_cross_sell, partnership_inquiry, technical_support,
     contract_negotiation, onboarding, cancellation_save, referral, general_inquiry, unknown
   - Secondary / latent intents the customer hinted at
   - Confidence score 0.0–1.0

2. OBJECTION IDENTIFICATION
   For each objection determine:
   - The objection text (paraphrased)
   - Category: price, budget, timing, authority, need, competitor, trust, complexity,
     contract_terms, integration, support, other
   - Severity: low / medium / high
   - Whether it was addressed during the call
   - A suggested rebuttal the rep could use in the follow-up

3. NEXT STEPS
   For every implicit or explicit next step:
   - The action description
   - Owner: sales_rep / customer / both / unknown
   - Priority: low / medium / high / critical
   - Deadline (if mentioned or inferable, else null)
   - Category: schedule_demo, send_pricing, send_proposal, send_contract,
     send_documentation, internal_review, follow_up_call, technical_evaluation,
     decision_meeting, trial_setup, onboarding, escalation, custom, none

4. KEY TOPICS — what was discussed and the status of each point

5. BUYING SIGNALS — phrases or behaviors indicating purchase likelihood
   - Strength: weak / moderate / strong

6. METADATA — customer_name, rep_name, company_mentioned, deal_stage, urgency_level,
   conversation_quality

TASK 2 — FOLLOW-UP EMAIL GENERATION

Using your analysis, compose a follow-up email that:

a) SUBJECT LINE
   - Specific, benefit-oriented, references a concrete topic from the call
   - Not generic ("Following up on our call")
   - Under 60 characters

b) GREETING — personalized with the customer's name if available

c) BODY
   - Opening: reference a specific moment, statement, or concern from the call to
     show active listening (1-2 sentences)
   - Value recap: summarize the key value proposition(s) discussed, tying them to the
     customer's stated needs or pain points (2-3 sentences)
   - Objection pre-handling: if unresolved objections exist, address the top one with
     a concise rebuttal, social proof, or reframe (1-2 sentences)
   - Deliverables: list any promised materials (pricing, proposal, docs) and confirm
     they are attached or will follow (1 sentence)

d) CALL TO ACTION
   - ONE clear, specific, low-friction CTA
   - Include a concrete time suggestion when scheduling
   - Examples: "Would Thursday at 2 PM work for a 30-minute demo?"
               "I've attached the pricing sheet — could you review and share any
                questions by Friday?"

e) CLOSING — professional sign-off with rep's name if available

=== EDGE CASES ===

Handle these gracefully:

- VOICEMAIL / NO CONVERSATION: Generate a brief "sorry I missed you" email with a
  scheduling CTA. Set data_sufficient=false.
- IVR / AUTOMATED SYSTEM ONLY: Set data_sufficient=false, generate a generic
  re-engagement email.
- ANGRY / VERY NEGATIVE CUSTOMER: Use empathetic tone, lead with acknowledgement of
  their concern, CTA should be low-pressure.
- CANCELLATION INTENT: Use a save-oriented email — highlight value, offer concessions
  or escalation.
- EXTREMELY SHORT TRANSCRIPT (<50 words): Set data_sufficient=false, generate a
  polite check-in email.
- NO CLEAR NEXT STEP: Suggest a reasonable default CTA based on inferred deal stage.
- MULTIPLE DECISION-MAKERS MENTIONED: Address the email to the primary contact while
  referencing the need to loop in others.
- COMPETITOR MENTIONED: Include a subtle differentiator in the email body.
- TECHNICAL QUESTIONS UNRESOLVED: CTA should offer a technical deep-dive or
  specialist call.
- CUSTOMER REQUESTED SPECIFIC MATERIALS: Explicitly mention those materials in the
  email and confirm delivery.

=== OUTPUT FORMAT ===

Return ONLY the following JSON. No additional text, no markdown, no explanation.

{{
    "follow_up_email": {{
        "subject": "...",
        "greeting": "...",
        "body": "...",
        "call_to_action": "...",
        "closing": "...",
        "full_email": "complete email text assembled from greeting + body + call_to_action + closing",
        "tone": "professional|friendly|urgent|consultative|empathetic|assertive",
        "email_type": "demo_follow_up|pricing_follow_up|proposal_follow_up|meeting_recap|objection_handling|next_steps_confirmation|thank_you|re_engagement|contract_follow_up|general_follow_up|problem_solving|information_follow_up"
    }},
    "data_sufficient": true/false
}}

IMPORTANT RULES:
- Return ONLY valid JSON, no surrounding text
- Use null for missing values, never omit fields
- Boolean values must be lowercase true/false
- Keep the email professional, specific, and under 250 words
- The call_to_action MUST contain a specific proposed time or concrete action
- Do NOT invent facts not present in the transcript
- If customer name is unknown, use a polite generic greeting
- full_email MUST be the complete email text ready to send (greeting + body + CTA + closing combined)
- Set data_sufficient to false ONLY when the transcript lacks enough content for a meaningful email"""

        full_prompt = self._format_prompt(user_prompt)
        return await self._generate_async(full_prompt, request_id)