#!/usr/bin/env python3
"""
Unit tests for task_generation_client.py
Tests Pydantic validation, prompt formatting, JSON preprocessing, and post-LLM injection.
No Triton server required — all gRPC calls are mocked.
"""

import json
import pytest
import sys
import os
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligent_pipeline.task_generation_client import (
    AgentTask,
    TaskGenerationResult,
    AsyncTaskGenerationClient,
)


# ===== Pydantic Model Tests =====

class TestAgentTask:
    """Test AgentTask Pydantic model validation and normalization."""

    def test_valid_task(self):
        task = AgentTask(
            task_type="callback",
            title="Call back customer",
            description="Customer requested callback regarding policy renewal",
            priority="high",
            due_date_relative="3 days",
            due_date_estimated="2026-02-15T00:00:00Z",
            assigned_to="1003-rti.talkloop.ai",
            name="John Smith",
            number="+16124571124",
            source_action_item="Customer asked to be called back in 3 days",
            confidence=0.9,
        )
        assert task.task_type == "callback"
        assert task.priority == "high"
        assert task.confidence == 0.9

    def test_defaults(self):
        task = AgentTask()
        assert task.task_type == "follow_up"
        assert task.priority == "medium"
        assert task.confidence == 0.5
        assert task.title == "No title"

    # -- task_type normalization --

    def test_task_type_callback_variations(self):
        assert AgentTask(task_type="call back").task_type == "callback"
        assert AgentTask(task_type="Call-Back").task_type == "callback"
        assert AgentTask(task_type="callback").task_type == "callback"

    def test_task_type_follow_up_variations(self):
        assert AgentTask(task_type="follow up").task_type == "follow_up"
        assert AgentTask(task_type="follow-up").task_type == "follow_up"
        assert AgentTask(task_type="Follow_Up").task_type == "follow_up"

    def test_task_type_escalation_variations(self):
        assert AgentTask(task_type="escalation").task_type == "escalation"
        assert AgentTask(task_type="escalate").task_type == "escalation"

    def test_task_type_documentation_variations(self):
        assert AgentTask(task_type="documentation").task_type == "documentation"
        assert AgentTask(task_type="document").task_type == "documentation"

    def test_task_type_null_defaults(self):
        assert AgentTask(task_type=None).task_type == "follow_up"
        assert AgentTask(task_type="null").task_type == "follow_up"
        assert AgentTask(task_type="N/A").task_type == "follow_up"

    def test_task_type_unknown_becomes_other(self):
        assert AgentTask(task_type="something_random").task_type == "other"

    # -- priority normalization --

    def test_priority_urgent(self):
        assert AgentTask(priority="urgent").priority == "urgent"
        assert AgentTask(priority="critical").priority == "urgent"
        assert AgentTask(priority="URGENT").priority == "urgent"

    def test_priority_high(self):
        assert AgentTask(priority="high").priority == "high"
        assert AgentTask(priority="HIGH").priority == "high"

    def test_priority_low(self):
        assert AgentTask(priority="low").priority == "low"

    def test_priority_default(self):
        assert AgentTask(priority=None).priority == "medium"
        assert AgentTask(priority="something").priority == "medium"

    # -- confidence coercion --

    def test_confidence_clamped_high(self):
        task = AgentTask(confidence=1.5)
        assert task.confidence == 1.0

    def test_confidence_clamped_low(self):
        task = AgentTask(confidence=-0.3)
        assert task.confidence == 0.0

    def test_confidence_string_input(self):
        task = AgentTask(confidence="0.85")
        assert task.confidence == 0.85

    def test_confidence_invalid_string(self):
        task = AgentTask(confidence="not_a_number")
        assert task.confidence == 0.5


class TestTaskGenerationResult:
    """Test TaskGenerationResult Pydantic model."""

    def test_empty_tasks_gets_fallback_reason(self):
        result = TaskGenerationResult(tasks=[])
        assert result.task_count == 0
        assert result.fallback_reason is not None

    def test_task_count_synced(self):
        result = TaskGenerationResult(
            tasks=[AgentTask(), AgentTask()],
            task_count=999,  # should be overridden
        )
        assert result.task_count == 2

    def test_with_tasks(self):
        result = TaskGenerationResult(
            tasks=[
                AgentTask(title="Task 1"),
                AgentTask(title="Task 2"),
            ],
            data_sufficient=True,
        )
        assert result.task_count == 2
        assert result.data_sufficient is True
        assert result.fallback_reason is None

    def test_explicit_fallback_reason_preserved(self):
        result = TaskGenerationResult(
            tasks=[],
            data_sufficient=False,
            fallback_reason="Custom reason",
        )
        assert result.fallback_reason == "Custom reason"


# ===== Client Unit Tests (mocked gRPC) =====

class TestAsyncTaskGenerationClient:
    """Test client logic with mocked Triton gRPC."""

    @pytest.fixture
    def client(self):
        """Create a client with mocked config."""
        with patch("intelligent_pipeline.task_generation_client.config") as mock_config:
            mock_config.get_service_config.return_value = {
                "url": "localhost:3801",
                "model_name": "qwen-task-gen",
            }
            return AsyncTaskGenerationClient()

    # -- prompt formatting --

    def test_format_prompt_qwen_template(self, client):
        prompt = client._AsyncTaskGenerationClient__format_prompt("Hello world")
        assert "<|im_start|>system" in prompt
        assert "<|im_end|>" in prompt
        assert "<|im_start|>user\nHello world<|im_end|>" in prompt
        assert "<|im_start|>assistant\n" in prompt

    def test_format_prompt_contains_system_prompt(self, client):
        prompt = client._AsyncTaskGenerationClient__format_prompt("test")
        assert "task generation system" in prompt or "call center" in prompt

    # -- JSON preprocessing --

    def test_preprocess_clean_json(self, client):
        raw = json.dumps({
            "tasks": [
                {
                    "task_type": "callback",
                    "title": "Call back",
                    "description": "desc",
                    "priority": "high",
                    "source_action_item": "item",
                    "confidence": 0.9,
                }
            ],
            "task_count": 1,
            "data_sufficient": True,
            "fallback_reason": None,
        })
        result = client._AsyncTaskGenerationClient__preprocess_output(raw)
        assert result is not None
        assert "tasks" in result
        assert len(result["tasks"]) == 1

    def test_preprocess_markdown_wrapped_json(self, client):
        raw = '```json\n{"tasks": [{"task_type": "follow_up", "title": "Test", "description": "d", "priority": "medium", "source_action_item": "a", "confidence": 0.8}], "task_count": 1, "data_sufficient": true}\n```'
        result = client._AsyncTaskGenerationClient__preprocess_output(raw)
        assert result is not None
        assert result["tasks"][0]["title"] == "Test"

    def test_preprocess_json_with_surrounding_text(self, client):
        raw = 'Here is the result:\n{"tasks": [{"task_type": "callback", "title": "CB", "description": "d", "priority": "high", "source_action_item": "a", "confidence": 0.9}], "task_count": 1, "data_sufficient": true}\nDone.'
        result = client._AsyncTaskGenerationClient__preprocess_output(raw)
        assert result is not None
        assert result["tasks"][0]["task_type"] == "callback"

    def test_preprocess_empty_string(self, client):
        result = client._AsyncTaskGenerationClient__preprocess_output("")
        assert result is None

    def test_preprocess_no_json(self, client):
        result = client._AsyncTaskGenerationClient__preprocess_output("This is just plain text with no JSON")
        assert result is None

    # -- clean_output end-to-end --

    def test_clean_output_valid(self, client):
        raw = json.dumps({
            "tasks": [
                {
                    "task_type": "callback",
                    "title": "Call customer",
                    "description": "Customer wants callback in 3 days",
                    "priority": "high",
                    "due_date_relative": "3 days",
                    "due_date_estimated": "2026-02-15T00:00:00Z",
                    "source_action_item": "Call back in 3 days",
                    "confidence": 0.9,
                }
            ],
            "task_count": 1,
            "data_sufficient": True,
        })
        result = client._AsyncTaskGenerationClient__clean_output(raw)
        assert result["data_sufficient"] is True
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["task_type"] == "callback"
        assert result["task_count"] == 1

    def test_clean_output_garbage(self, client):
        result = client._AsyncTaskGenerationClient__clean_output("not json at all !!!")
        assert result["data_sufficient"] is False
        assert result["fallback_reason"] is not None
        assert result["tasks"] == []

    # -- post-LLM injection --

    @pytest.mark.asyncio
    async def test_post_llm_injection(self, client):
        """Test that assigned_to, name, number are injected from metadata after LLM response."""
        mock_response = MagicMock()
        mock_response.as_numpy.return_value = np.array(
            [json.dumps({
                "tasks": [
                    {
                        "task_type": "callback",
                        "title": "Call customer",
                        "description": "Follow up on issue",
                        "priority": "medium",
                        "source_action_item": "Call back customer",
                        "confidence": 0.85,
                        "assigned_to": "wrong_agent",
                        "name": "wrong_name",
                        "number": "wrong_number",
                    }
                ],
                "task_count": 1,
                "data_sufficient": True,
            }).encode("utf-8")],
            dtype=object,
        )

        client.client = AsyncMock()
        client.client.infer = AsyncMock(return_value=mock_response)

        result = await client.generate_tasks(
            action_items=[{"description": "Call back customer", "owner": "Agent"}],
            call_metadata={
                "src": "+16124571124",
                "agentExtension": "1003-rti.talkloop.ai",
                "calldate": "2026-02-10 12:14:30",
            },
            contact_info={"name": "John Smith"},
            request_id="test-001",
        )

        tasks = result["agent_tasks"]["tasks"]
        assert len(tasks) == 1
        # Post-LLM injection should override LLM values
        assert tasks[0]["assigned_to"] == "1003-rti.talkloop.ai"
        assert tasks[0]["name"] == "John Smith"
        assert tasks[0]["number"] == "+16124571124"

    @pytest.mark.asyncio
    async def test_empty_action_items_returns_fallback(self, client):
        """Empty action items should return a fallback without calling Triton."""
        client.client = AsyncMock()

        result = await client.generate_tasks(
            action_items=[],
            call_metadata={"src": "+1234567890"},
            contact_info={},
            request_id="test-empty",
        )

        assert result["agent_tasks"]["data_sufficient"] is False
        assert "No action items" in result["agent_tasks"]["fallback_reason"]
        # Should NOT have called Triton
        client.client.infer.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_tasks_triton_error(self, client):
        """Test that Triton errors are handled gracefully by the caller (pipeline)."""
        client.client = AsyncMock()
        client.client.infer = AsyncMock(side_effect=Exception("Triton connection refused"))

        with pytest.raises(Exception, match="Triton connection refused"):
            await client.generate_tasks(
                action_items=[{"description": "Follow up", "owner": "Agent"}],
                call_metadata={"src": "+1234"},
                contact_info={},
                request_id="test-err",
            )


# ===== Pipeline Integration Tests (mocked) =====

class TestPipelineTaskGeneration:
    """Test task generation wiring in the pipeline."""

    def test_call_metadata_schema_validation(self):
        """Test CallMetadata Pydantic model accepts valid CDR."""
        from api.schemas.transcribe import CallMetadata

        metadata = CallMetadata(
            uniqueId="1770743670.1246",
            src="+16124571124",
            dst="8888090229",
            accountId="5IR02639",
            disposition="ANSWERED",
            calldate="2026-02-10 12:14:30",
            agentExtension="1003-rti.talkloop.ai",
            direction="OUTBOUND",
        )
        assert metadata.src == "+16124571124"
        assert metadata.direction == "OUTBOUND"

    def test_call_metadata_all_optional(self):
        """All fields are optional — empty CDR should be valid."""
        from api.schemas.transcribe import CallMetadata

        metadata = CallMetadata()
        assert metadata.src is None
        assert metadata.calldate is None

    def test_webhook_payload_accepts_agent_tasks(self):
        """WebhookPayload dataclass should accept optional agent_tasks field."""
        from webhook import WebhookPayload

        payload = WebhookPayload(
            job_id="test-uuid",
            transcription="Hello",
            agent_tasks={
                "tasks": [{"task_type": "callback", "title": "Test"}],
                "task_count": 1,
                "data_sufficient": True,
            },
        )
        assert payload.agent_tasks is not None
        assert payload.agent_tasks["task_count"] == 1

    def test_webhook_payload_agent_tasks_optional(self):
        """agent_tasks should be optional (default None) in webhook payload."""
        from webhook import WebhookPayload

        payload = WebhookPayload(
            job_id="test-uuid",
            transcription="Hello",
        )
        assert payload.agent_tasks is None
