import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal
import numpy as np
import tritonclient.grpc.aio as grpcclient_aio
from pydantic import BaseModel, Field, field_validator, model_validator
from .config_loader import config


# ===== Pydantic Models =====

class AgentTask(BaseModel):
    """A single actionable task generated from call analysis."""
    task_type: Literal["callback", "follow_up", "escalation", "documentation", "other"] = "follow_up"
    title: str = "No title"
    description: str = "No description"
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    due_date_relative: Optional[str] = None
    due_date_estimated: Optional[str] = None
    assigned_to: Optional[str] = None
    name: Optional[str] = None
    number: Optional[str] = None
    source_action_item: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator('confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (ValueError, TypeError):
            return 0.5

    @field_validator('task_type', mode='before')
    @classmethod
    def normalize_task_type(cls, v):
        if not v or str(v).lower() in ['null', 'none', 'n/a']:
            return "follow_up"
        normalized = str(v).lower().replace(" ", "_").replace("-", "_")
        valid = {"callback", "follow_up", "escalation", "documentation", "other"}
        # Map common variations
        if "call" in normalized and "back" in normalized:
            return "callback"
        if "follow" in normalized:
            return "follow_up"
        if "escalat" in normalized:
            return "escalation"
        if "doc" in normalized:
            return "documentation"
        return normalized if normalized in valid else "other"

    @field_validator('priority', mode='before')
    @classmethod
    def normalize_priority(cls, v):
        if not v:
            return "medium"
        v_str = str(v).lower()
        if "urgent" in v_str or "critical" in v_str:
            return "urgent"
        if "high" in v_str:
            return "high"
        if "low" in v_str:
            return "low"
        return "medium"


class TaskGenerationResult(BaseModel):
    """Result of task generation from call analysis."""
    tasks: List[AgentTask] = Field(default_factory=list)
    task_count: int = 0
    data_sufficient: bool = True
    fallback_reason: Optional[str] = None

    @model_validator(mode='after')
    def sync_task_count(self):
        self.task_count = len(self.tasks)
        if not self.tasks and not self.fallback_reason:
            self.fallback_reason = "No actionable tasks identified from the call"
        return self


class AsyncTaskGenerationClient:
    """Async gRPC Client for Triton Qwen Task Generation Server"""

    def __init__(self, url: str = None):
        service_config = config.get_service_config('task_generation')

        self.url = url or service_config.get('url', 'localhost:3801')
        self.model_name = service_config.get('model_name', 'qwen-task-gen')
        self.client = None
        self.system_prompt = """You are a call center task generation system. You convert action items from customer service calls into structured JSON tasks.

RULES:
1. Every action item that mentions an action (call back, send, follow up, escalate, update, submit, schedule) MUST become a task. When in doubt, create the task.
2. "escalate" or "supervisor" → task_type: "escalation", priority: "high" or "urgent"
3. "call back" or "callback" → task_type: "callback"
4. Any other follow-up action → task_type: "follow_up"
5. Sending documents or updating records → task_type: "documentation"
6. If no time reference is mentioned, default due_date_relative to "1 business day"
7. Calculate due_date_estimated as ISO date from calldate + due_date_relative
8. EVERY task MUST include a "confidence" field (float 0.0-1.0). Never omit it.
9. Only return data_sufficient=false if action items are truly empty or say "no action required"
10. Even with minimal metadata, still create tasks — missing metadata is not a reason to skip tasks

Return ONLY valid JSON. No extra text."""

    async def connect(self):
        """Connect to Triton server"""
        self.client = grpcclient_aio.InferenceServerClient(url=self.url)

        if not await self.client.is_server_live():
            raise Exception(f"Triton server at {self.url} is not live")

        if not await self.client.is_server_ready():
            raise Exception(f"Triton server at {self.url} is not ready")

        if not await self.client.is_model_ready(self.model_name):
            raise Exception(f"Model {self.model_name} is not ready")

        print(f"[Task Generation] Connected to Triton server at {self.url}")

    def __format_prompt(self, user_prompt: str) -> str:
        """Format prompt using Qwen chat template."""
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

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

            # Strategy 1: Find JSON objects containing "tasks"
            json_pattern = r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}[^{}]*)*\}'

            potential_jsons = []
            for match in re.finditer(json_pattern, output_text):
                json_str = match.group(0)
                if '"tasks"' in json_str:
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and 'tasks' in parsed:
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
                        if '"tasks"' in json_str:
                            json_starts.append((start, i+1, json_str))

            for start, end, json_str in reversed(json_starts):
                try:
                    parsed = json.loads(json_str)
                    if 'tasks' in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue

            return None

        except Exception as e:
            print(f"[Task Generation] Preprocessing error: {e}")
            return None

    def __postprocess_with_pydantic(self, raw_dict: Dict) -> Dict:
        """Post-process using Pydantic models for validation and cleaning."""
        try:
            result = TaskGenerationResult(**raw_dict)
            return json.loads(result.model_dump_json())
        except Exception as e:
            print(f"[Task Generation] Pydantic validation error: {e}")
            return self.__get_default_response()

    def __get_default_response(self) -> Dict:
        """Return safe default response."""
        return json.loads(TaskGenerationResult(
            data_sufficient=False,
            fallback_reason="Failed to parse task generation output"
        ).model_dump_json())

    def __clean_output(self, output_text: str) -> Dict:
        """Clean and format the output to ensure consistent structure."""
        try:
            raw_result = self.__preprocess_output(output_text)
            if raw_result:
                return self.__postprocess_with_pydantic(raw_result)

            print(f"[Task Generation] Preprocessing failed")
            print(f"[Task Generation] Output text length: {len(output_text)}")
            print(f"[Task Generation] First 200 chars: {output_text[:200]}")

            return self.__get_default_response()

        except Exception as e:
            print(f"[Task Generation] Unexpected error in __clean_output: {e}")
            return self.__get_default_response()

    async def generate_tasks(
        self,
        action_items: List[Dict],
        call_metadata: Dict,
        contact_info: Dict,
        request_id: str
    ) -> Dict:
        """
        Generate agent tasks from action items, call metadata, and contact info.

        Args:
            action_items: List of action items from analysis_client (Mistral-Nemo)
                          e.g., [{"description": "...", "owner": "Agent"}]
            call_metadata: CDR fields from Talkloop
                           e.g., {"uniqueId": "...", "src": "+16124571124",
                                  "dst": "8888090229", "accountId": "...",
                                  "disposition": "ANSWERED", "calldate": "...",
                                  "agentExtension": "1003-rti.talkloop.ai",
                                  "direction": "OUTBOUND"}
            contact_info: Contact info extracted by Mistral-Nemo from transcript
                          e.g., {"name": "John Smith", "phone": "+1234567890", "email": "..."}
            request_id: Unique request identifier

        Returns:
            Dict with task generation results
        """
        # Handle empty action items
        if not action_items:
            print(f"[Task Generation] No action items provided for request_id: {request_id}")
            return {
                "agent_tasks": json.loads(TaskGenerationResult(
                    data_sufficient=False,
                    fallback_reason="No action items provided from call analysis"
                ).model_dump_json())
            }

        if not self.client:
            await self.connect()

        # Format action items for the prompt
        action_items_text = "\n".join(
            f"- {item.get('description', 'Unknown')} (Owner: {item.get('owner', 'Unknown')})"
            for item in action_items
        )

        # Format call metadata for the prompt
        metadata_text = "\n".join(
            f"- {key}: {value}"
            for key, value in call_metadata.items()
            if value is not None
        )

        # Format contact info from Mistral-Nemo analysis
        contact_name = contact_info.get('name') or None
        contact_phone = call_metadata.get('src') or contact_info.get('phone') or None
        agent_extension = call_metadata.get('agentExtension') or None

        contact_text = ""
        if contact_name:
            contact_text += f"- Customer Name: {contact_name}\n"
        if contact_phone:
            contact_text += f"- Customer Phone: {contact_phone}\n"

        user_prompt = f"""Analyze the following action items from a customer service call and generate structured agent tasks.

## Call Metadata:
{metadata_text if metadata_text else "No metadata available"}

## Customer Contact Info (extracted from call):
{contact_text if contact_text else "No contact info available"}

## Action Items from Call:
{action_items_text}

## Instructions:
For each action item that requires agent follow-up, create a task. Use the call metadata (especially calldate) to calculate due dates.

Return ONLY a valid JSON object with no additional text, following this exact structure:

{{
  "tasks": [
    {{
      "task_type": "callback|follow_up|escalation|documentation|other",
      "title": "Short task title",
      "description": "Detailed description of what the agent needs to do",
      "priority": "low|medium|high|urgent",
      "due_date_relative": "e.g., 3 days, next week, tomorrow",
      "due_date_estimated": "ISO date string e.g., 2026-02-13T00:00:00Z",
      "assigned_to": "the agentExtension from call metadata if available, otherwise null",
      "name": "customer name from contact info if available, otherwise null",
      "number": "customer phone number (src from metadata) if available, otherwise null",
      "source_action_item": "The original action item this task was derived from",
      "confidence": 0.0-1.0
    }}
  ],
  "task_count": <number of tasks>,
  "data_sufficient": true/false,
  "fallback_reason": null or "explanation if no tasks could be generated"
}}

IMPORTANT:
- Return ONLY valid JSON with no additional text
- Only create tasks for items that genuinely require follow-up action
- Calculate due_date_estimated from calldate + due_date_relative
- If calldate is not available, omit due_date_estimated
- For "assigned_to", always use the agentExtension value: "{agent_extension or 'null'}"
- For "name", use the customer name: "{contact_name or 'null'}"
- For "number", use the customer phone (src): "{contact_phone or 'null'}"
- Set confidence based on how clearly the action item implies a task"""

        prompt = self.__format_prompt(user_prompt)

        print(f"[Task Generation] Sending task generation request request_id: {request_id}")
        print(f"[Task Generation] Prompt length (chars): {len(prompt)}")
        print(f"[Task Generation] Action items count: {len(action_items)}")

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
        print(f"[Task Generation] Raw output shape: {raw_output.shape}")
        print(f"[Task Generation] Raw output dtype: {raw_output.dtype}")

        # Handle different output shapes
        if raw_output.ndim == 2:
            output_text = raw_output[0][0]
        else:
            output_text = raw_output[0]

        if isinstance(output_text, bytes):
            output_text = output_text.decode('utf-8')

        print(f"[Task Generation] Received response for request_id: {request_id}")
        print(f"[Task Generation] Output text length: {len(output_text)}")

        if len(output_text) > 0:
            print(f"[Task Generation] First 500 chars of response: {output_text[:500]}")

        # Parse and validate response
        result = self.__clean_output(output_text)

        # Inject known fields from metadata/contact_info (don't rely on LLM to copy these)
        contact_name = contact_info.get('name') or None
        contact_phone = call_metadata.get('src') or contact_info.get('phone') or None
        agent_extension = call_metadata.get('agentExtension') or None

        for task in result.get('tasks', []):
            if agent_extension:
                task['assigned_to'] = agent_extension
            if contact_name:
                task['name'] = contact_name
            if contact_phone:
                task['number'] = contact_phone

        return {"agent_tasks": result}

    async def close(self):
        """Close client connection"""
        if self.client:
            await self.client.close()
            print("[Task Generation] Connection closed")
