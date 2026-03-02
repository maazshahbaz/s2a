import tritonclient.grpc.aio as grpcclient_aio
import json
import numpy as np
import uuid
import re
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator
from config_loader import config


# --- Pydantic Models ---

class KeyUpdate(BaseModel):
    point: str = ""

    @field_validator("point", mode="before")
    @classmethod
    def clean_point(cls, v):
        if not v or not str(v).strip():
            return ""
        return str(v).strip()


class AgentSummaryResult(BaseModel):
    total_calls_reviewed: int = 0
    key_updates: List[KeyUpdate] = Field(default_factory=list)

    @field_validator("key_updates", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if not isinstance(v, list):
            return []
        return v


class AgentSummaryResponse(BaseModel):
    request_id: str
    success: bool
    result: Optional[AgentSummaryResult] = None
    error: Optional[str] = None


# --- Main Class ---

class AsyncAgentSummary:
    def __init__(self):
        service_config = config.get_service_config("analysis")
        self.url = service_config.get("url", "localhost:3701")
        self.model_name = service_config.get("model_name", "mistral-nemo")
        self.client = None

    async def initialize(self):
        if self.client is None:
            self.client = grpcclient_aio.InferenceServerClient(url=self.url)

    def __format_prompt(self, user_prompt: str) -> str:
        system_prompt = (
            "You are an expert call center manager assistant. "
            "You summarize multiple call summaries for a single agent into "
            "concise key updates that a manager can quickly review. "
            "You always respond with valid JSON only, no extra text."
        )
        return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"

    def __extract_json(self, output_text: str) -> dict | None:
        """Extract the first valid JSON object from LLM output."""
        output_text = re.sub(r"```json\s*", "", output_text)
        output_text = re.sub(r"```\s*", "", output_text)
        output_text = output_text.strip()

        try:
            parsed = json.loads(output_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        depth = 0
        start_pos = -1
        for i, char in enumerate(output_text):
            if char == "{":
                if depth == 0:
                    start_pos = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start_pos != -1:
                    try:
                        return json.loads(output_text[start_pos : i + 1])
                    except json.JSONDecodeError:
                        start_pos = -1

        return None

    def __build_response(
        self, raw: dict | None, total_calls: int, request_id: str
    ) -> AgentSummaryResponse:
        """Validate raw LLM output and return a clean response object."""
        if raw is None:
            return AgentSummaryResponse(
                request_id=request_id,
                success=False,
                error="Could not extract valid JSON from model output",
            )

        raw_updates = raw.get("key_updates", [])
        if not isinstance(raw_updates, list):
            raw_updates = []

        updates: list[KeyUpdate] = []
        for item in raw_updates:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = (item.get("point") or item.get("description") or "").strip()
            else:
                continue
            if text:
                updates.append(KeyUpdate(point=text))

        return AgentSummaryResponse(
            request_id=request_id,
            success=True,
            result=AgentSummaryResult(
                total_calls_reviewed=total_calls,
                key_updates=updates,
            ),
        )

    async def __generate_async(self, prompt: str, request_id: str) -> str:
        await self.initialize()

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

        output_text = response.as_numpy("generated_text")[0]
        if isinstance(output_text, bytes):
            output_text = output_text.decode("utf-8")
        return output_text

    async def generate_agent_summary(
        self,
        summaries: Dict[str, str],
        request_id: str | None = None,
    ) -> AgentSummaryResponse:
        """
        Generate 4-5 key update points from multiple call summaries.

        Args:
            summaries:  Dict like {"summary_1": "...", "summary_2": "...", ...}
            request_id: Optional tracking id.

        Returns:
            AgentSummaryResponse object.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        if not summaries:
            return AgentSummaryResponse(
                request_id=request_id,
                success=False,
                error="No summaries provided",
            )

        total_calls = len(summaries)

        summary_lines = "\n".join(
            f"{i+1}. {text}" for i, text in enumerate(summaries.values()) if text
        )

        user_prompt = f"""Below are {total_calls} call summaries handled by an agent.
Read ALL summaries carefully and produce exactly 4 to 5 key update points that a manager can review at a glance.
Each point should capture a distinct theme, recurring pattern, or notable event across the calls.

Call Summaries:
{summary_lines}

Return ONLY valid JSON in this exact structure (no extra text):
{{
    "key_updates": [
        "First key update point",
        "Second key update point",
        "Third key update point",
        "Fourth key update point",
        "Fifth key update point"
    ]
}}

IMPORTANT:
- Return ONLY the JSON, no markdown, no explanation
- Exactly 4 or 5 points, each a single concise sentence
- Cover the most important themes across all calls
- Be specific, not generic"""

        full_prompt = self.__format_prompt(user_prompt)

        try:
            output_text = await self.__generate_async(full_prompt, request_id)
            raw = self.__extract_json(output_text)
            return self.__build_response(raw, total_calls, request_id)
        except Exception as e:
            print(f"Error generating agent summary: {e}")
            return AgentSummaryResponse(
                request_id=request_id,
                success=False,
                error=str(e),
            )