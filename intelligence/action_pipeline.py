#!/usr/bin/env python3
"""
Action Pipeline: Schema-first extractor for meeting transcripts using Qwen2.5-7B-Instruct via vLLM
"""

import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

import httpx
from pydantic import BaseModel, Field, ValidationError


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Intent(str, Enum):
    PURCHASE_ORDER = "purchase_order"
    INVOICE_PROCESSING = "invoice_processing"
    MEETING_FOLLOW_UP = "meeting_follow_up"
    PROJECT_UPDATE = "project_update"
    CUSTOMER_SUPPORT = "customer_support"
    GENERAL_DISCUSSION = "general_discussion"


class ActionItem(BaseModel):
    assignee: Optional[str] = Field(None, description="Person assigned to the task")
    task: str = Field(..., description="Description of the action item")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format")
    priority: Priority = Field(Priority.MEDIUM, description="Task priority level")
    confidence: float = Field(0.8, description="Confidence score 0-1", ge=0, le=1)
    span: Optional[str] = Field(None, description="Relevant text span from transcript")


class Product(BaseModel):
    name: str = Field(..., description="Product name")
    quantity: Optional[int] = Field(None, description="Product quantity")
    price: Optional[float] = Field(None, description="Product price")


class Entity(BaseModel):
    invoice_ids: List[str] = Field(default_factory=list, description="Invoice identifiers")
    order_ids: List[str] = Field(default_factory=list, description="Order identifiers")
    products: List[Product] = Field(default_factory=list, description="Products mentioned")
    emails: List[str] = Field(default_factory=list, description="Email addresses")
    phones: List[str] = Field(default_factory=list, description="Phone numbers")
    money_amounts: List[float] = Field(default_factory=list, description="Monetary amounts")
    dates: List[str] = Field(default_factory=list, description="Dates in YYYY-MM-DD format")


class UnifiedPayload(BaseModel):
    summary: str = Field(..., description="Brief summary of the transcript")
    action_items: List[ActionItem] = Field(default_factory=list, description="Extracted action items")
    entities: Entity = Field(default_factory=Entity, description="Named entities")
    intent: Intent = Field(Intent.GENERAL_DISCUSSION, description="Primary intent classification")
    sentiment: Sentiment = Field(Sentiment.NEUTRAL, description="Overall sentiment")
    confidence_score: float = Field(0.8, description="Overall extraction confidence", ge=0, le=1)


class ExtractionMetrics(BaseModel):
    total_requests: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    retry_count: int = 0
    total_latency: float = 0.0
    avg_latency: float = 0.0
    json_pass_rate: float = 0.0
    field_hit_rates: Dict[str, float] = Field(default_factory=dict)


class ActionPipelineExtractor:
    """Schema-first extractor for meeting transcripts using vLLM + Qwen2.5-7B-Instruct"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
        self.metrics = ExtractionMetrics()
        
        # Compact schema for prompt
        self.schema_str = json.dumps(UnifiedPayload.model_json_schema(), separators=(',', ':'))
        
        # System prompt optimized for JSON output
        self.system_prompt = "Return ONLY valid JSON matching json_schema; unknown → null/[]"
        
        # Lightweight regex hints for better extraction
        self.regex_hints = {
            "invoice": r"(?i)inv(?:oice)?[-#\s]*([A-Z0-9-]+)",
            "order": r"(?i)(?:order|po)[-#\s]*([A-Z0-9-]+)",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"(?:\+1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})",
            "money": r"\$\s*([0-9,]+\.?[0-9]*)",
            "date": r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        }

    def filter_salient_lines(self, transcript: str) -> str:
        """Extract salient lines containing business-relevant information"""
        lines = transcript.split('\n')
        salient_keywords = [
            'invoice', 'order', 'purchase', 'payment', 'due', 'amount', 'price',
            'quantity', 'deliver', 'ship', 'follow up', 'action', 'task',
            'assign', 'responsible', 'deadline', 'priority', 'meeting',
            '@', 'email', 'phone', 'contact', '$', 'USD', 'total'
        ]
        
        salient_lines = []
        for line in lines:
            line = line.strip()
            if line and any(keyword.lower() in line.lower() for keyword in salient_keywords):
                salient_lines.append(line)
        
        # If no salient lines found, return first 10 lines
        if not salient_lines:
            return '\n'.join(lines[:10])
            
        return '\n'.join(salient_lines[:20])  # Limit to 20 most relevant lines

    def build_prompt(self, transcript: str) -> List[Dict[str, str]]:
        """Build prompt with system message and embedded schema"""
        salient_text = self.filter_salient_lines(transcript)
        
        # Add regex hints to help extraction
        hint_text = "Look for: invoices (INV-123), orders (PO-456), emails, phones, dates (YYYY-MM-DD), money ($123.45)"
        
        user_message = f"{hint_text}\n\nTranscript:\n{salient_text}"
        
        return [
            {"role": "system", "content": f"{self.system_prompt}\n\nSchema: {self.schema_str}"},
            {"role": "user", "content": user_message}
        ]

    def extract(self, transcript: str) -> Dict[str, Any]:
        """Extract structured data from transcript with retry logic"""
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            # Build initial prompt
            messages = self.build_prompt(transcript)
            
            # Call OpenAI-compatible API
            response = self._call_api(messages)
            
            try:
                # Validate with Pydantic
                payload = UnifiedPayload.model_validate(response)
                
                # Update metrics
                self.metrics.successful_extractions += 1
                self._update_field_hit_rates(payload)
                
                result = {
                    "success": True,
                    "data": payload.model_dump(),
                    "error": None,
                    "latency": time.time() - start_time
                }
                
            except ValidationError as e:
                # Retry once with error feedback
                self.metrics.retry_count += 1
                retry_result = self._retry_with_feedback(messages, str(e), start_time)
                if retry_result:
                    return retry_result
                
                # Return partial results on validation failure
                result = {
                    "success": False,
                    "data": response if isinstance(response, dict) else {},
                    "error": f"Validation error: {str(e)}",
                    "latency": time.time() - start_time
                }
                self.metrics.failed_extractions += 1
                
        except Exception as e:
            # Handle API or other errors
            result = {
                "success": False,
                "data": {},
                "error": f"Extraction error: {str(e)}",
                "latency": time.time() - start_time
            }
            self.metrics.failed_extractions += 1
        
        # Update latency metrics
        self._update_latency_metrics(result["latency"])
        
        return result

    def _call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call the vLLM OpenAI-compatible API"""
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 500,
            "response_format": {"type": "json_object"}
        }
        
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        return json.loads(content)

    def _retry_with_feedback(self, original_messages: List[Dict[str, str]], 
                           error_msg: str, start_time: float) -> Optional[Dict[str, Any]]:
        """Retry extraction with validation error feedback for self-correction"""
        retry_messages = original_messages + [
            {"role": "assistant", "content": "I'll provide valid JSON."},
            {"role": "user", "content": f"Previous response had validation error: {error_msg}. Please fix and return valid JSON only."}
        ]
        
        try:
            response = self._call_api(retry_messages)
            payload = UnifiedPayload.model_validate(response)
            
            self.metrics.successful_extractions += 1
            self._update_field_hit_rates(payload)
            
            return {
                "success": True,
                "data": payload.model_dump(),
                "error": "Recovered after retry",
                "latency": time.time() - start_time
            }
            
        except (ValidationError, Exception):
            return None

    def _update_field_hit_rates(self, payload: UnifiedPayload):
        """Update field hit rate metrics"""
        hits = {
            "summary": bool(payload.summary),
            "action_items": bool(payload.action_items),
            "invoice_ids": bool(payload.entities.invoice_ids),
            "order_ids": bool(payload.entities.order_ids),
            "products": bool(payload.entities.products),
            "emails": bool(payload.entities.emails),
            "phones": bool(payload.entities.phones),
            "money_amounts": bool(payload.entities.money_amounts),
            "dates": bool(payload.entities.dates)
        }
        
        for field, hit in hits.items():
            if field not in self.metrics.field_hit_rates:
                self.metrics.field_hit_rates[field] = 0.0
            
            # Simple moving average
            current_rate = self.metrics.field_hit_rates[field]
            new_rate = (current_rate * (self.metrics.total_requests - 1) + (1.0 if hit else 0.0)) / self.metrics.total_requests
            self.metrics.field_hit_rates[field] = new_rate

    def _update_latency_metrics(self, latency: float):
        """Update latency metrics"""
        self.metrics.total_latency += latency
        self.metrics.avg_latency = self.metrics.total_latency / self.metrics.total_requests
        
        # Update JSON pass rate
        if self.metrics.total_requests > 0:
            self.metrics.json_pass_rate = self.metrics.successful_extractions / self.metrics.total_requests

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.model_dump()

    def close(self):
        """Close HTTP client"""
        self.client.close()


# Example usage and testing
if __name__ == "__main__":
    # Sample transcript for testing
    sample_transcript = """
    [Meeting Recording 2024-01-15]
    
    John: Thanks everyone for joining today's procurement review. 
    Sarah: We need to process invoice INV-2024-001 for $15,000 from Acme Corp.
    Mike: I'll handle that by Friday. Also, PO-2024-045 needs approval for 100 units of Widget A at $25 each.
    Sarah: Can you send the details to sarah@company.com?
    John: Sure. The customer at 555-123-4567 called about their delivery on 2024-01-20.
    Mike: I'll follow up with them tomorrow and update the order status.
    Sarah: Let's also review the Q1 budget - we're looking at approximately $250,000 total.
    John: Action items: Mike handles invoice processing, Sarah approves PO, I'll call customer.
    """
    
    # Initialize extractor
    extractor = ActionPipelineExtractor()
    
    try:
        # Extract structured data
        result = extractor.extract(sample_transcript)
        
        print("Extraction Result:")
        print(json.dumps(result, indent=2))
        
        print("\nPerformance Metrics:")
        print(json.dumps(extractor.get_metrics(), indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        extractor.close()