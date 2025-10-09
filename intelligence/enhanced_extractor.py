#!/usr/bin/env python3
"""
Enhanced Business Intelligence Extractor for S2A Pipeline
Comprehensive extraction using the enhanced schema for sales, customer support, and general business contexts
"""

import json
import time
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Type
from enum import Enum

import httpx
from pydantic import BaseModel, Field, ValidationError

from .enhanced_schema import (
    EnhancedBusinessIntelligence, SalesIntelligence, SupportIntelligence,
    CallType, Intent, Sentiment, Priority, CustomerStage, IssueStatus
)


class ExtractionMode(str, Enum):
    GENERAL = "general"
    SALES = "sales"
    SUPPORT = "support"
    AUTO_DETECT = "auto_detect"


class ConversationAnalyzer:
    """Analyzes conversation patterns and metrics"""

    def __init__(self):
        self.speaker_patterns = [
            r"([A-Z][a-z]+):\s*",  # "John: "
            r"([A-Z][a-z]+)\s*-\s*",  # "John - "
            r"\[([A-Z][a-z]+)\]",  # "[John]"
            r"<([A-Z][a-z]+)>",  # "<John>"
        ]

    def extract_speakers(self, transcript: str) -> List[str]:
        """Extract speaker names from transcript"""
        speakers = set()
        for pattern in self.speaker_patterns:
            matches = re.findall(pattern, transcript)
            speakers.update(matches)
        return list(speakers)

    def analyze_talk_time(self, transcript: str) -> Dict[str, float]:
        """Analyze talk time distribution (basic estimation)"""
        speakers = self.extract_speakers(transcript)
        if len(speakers) < 2:
            return {}

        lines = transcript.split('\n')
        speaker_lines = {}

        for line in lines:
            for pattern in self.speaker_patterns:
                match = re.match(pattern, line)
                if match:
                    speaker = match.group(1)
                    if speaker not in speaker_lines:
                        speaker_lines[speaker] = 0
                    speaker_lines[speaker] += len(line)
                    break

        total_content = sum(speaker_lines.values())
        if total_content == 0:
            return {}

        # Convert to percentages
        talk_time_percent = {}
        for speaker, content_length in speaker_lines.items():
            talk_time_percent[speaker] = (content_length / total_content) * 100

        return talk_time_percent

    def count_questions(self, transcript: str) -> Dict[str, int]:
        """Count questions in the conversation"""
        lines = transcript.split('\n')
        total_questions = 0
        speaker_questions = {}

        for line in lines:
            question_count = line.count('?')
            total_questions += question_count

            # Try to attribute to speaker
            for pattern in self.speaker_patterns:
                match = re.match(pattern, line)
                if match:
                    speaker = match.group(1)
                    if speaker not in speaker_questions:
                        speaker_questions[speaker] = 0
                    speaker_questions[speaker] += question_count
                    break

        return {
            "total": total_questions,
            "by_speaker": speaker_questions
        }

    def detect_interruptions(self, transcript: str) -> int:
        """Detect potential interruptions (basic heuristic)"""
        # Look for patterns like "--", sudden speaker changes, etc.
        interruption_patterns = [
            r"--",
            r"\[interrupting\]",
            r"\[cuts off\]",
            r"sorry to interrupt",
            r"let me stop you there"
        ]

        interruptions = 0
        for pattern in interruption_patterns:
            interruptions += len(re.findall(pattern, transcript, re.IGNORECASE))

        return interruptions


class EnhancedExtractor:
    """Enhanced business intelligence extractor with comprehensive schema support"""

    def __init__(self, base_url: str = "http://localhost:8000/v1", mode: ExtractionMode = ExtractionMode.AUTO_DETECT):
        self.base_url = base_url
        self.mode = mode
        self.client = httpx.Client(timeout=45.0)  # Longer timeout for complex extractions
        self.conversation_analyzer = ConversationAnalyzer()

        # Metrics tracking
        self.metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "avg_latency": 0.0,
            "mode_usage": {mode.value: 0 for mode in ExtractionMode}
        }

    def auto_detect_mode(self, transcript: str) -> ExtractionMode:
        """Auto-detect the conversation type from transcript content"""
        text_lower = transcript.lower()

        # Sales indicators
        sales_keywords = [
            "price", "cost", "budget", "quote", "proposal", "contract", "deal",
            "purchase", "buy", "sell", "roi", "revenue", "pricing", "discount",
            "demo", "trial", "competitor", "features", "benefits"
        ]

        # Support indicators
        support_keywords = [
            "issue", "problem", "error", "bug", "help", "support", "ticket",
            "troubleshoot", "fix", "resolve", "broken", "not working", "crash",
            "escalate", "refund", "complaint", "billing issue"
        ]

        sales_score = sum(1 for keyword in sales_keywords if keyword in text_lower)
        support_score = sum(1 for keyword in support_keywords if keyword in text_lower)

        if sales_score > support_score and sales_score >= 3:
            return ExtractionMode.SALES
        elif support_score > sales_score and support_score >= 3:
            return ExtractionMode.SUPPORT
        else:
            return ExtractionMode.GENERAL

    def get_schema_class(self, mode: ExtractionMode) -> Type[BaseModel]:
        """Get the appropriate schema class for the mode"""
        if mode == ExtractionMode.SALES:
            return SalesIntelligence
        elif mode == ExtractionMode.SUPPORT:
            return SupportIntelligence
        else:
            return EnhancedBusinessIntelligence

    def build_enhanced_prompt(self, transcript: str, mode: ExtractionMode) -> List[Dict[str, str]]:
        """Build comprehensive prompt for enhanced extraction"""

        # Get schema for the mode
        schema_class = self.get_schema_class(mode)
        schema_str = json.dumps(schema_class.model_json_schema(), separators=(',', ':'))

        # Mode-specific instructions
        mode_instructions = {
            ExtractionMode.SALES: """
Focus on sales-specific elements:
- Lead qualification and buying signals
- Product interest and objections
- Budget discussions and pricing
- Decision makers and timeline
- Competitive mentions
- Next steps and commitments
""",
            ExtractionMode.SUPPORT: """
Focus on support-specific elements:
- Customer issues and problems
- Resolution steps and timeline
- Customer satisfaction signals
- Escalation risks
- Knowledge gaps
- Technical details
""",
            ExtractionMode.GENERAL: """
Focus on general business elements:
- Action items and responsibilities
- Key decisions and outcomes
- Important entities and contacts
- Follow-up requirements
"""
        }

        # Extract key conversation metrics
        speakers = self.conversation_analyzer.extract_speakers(transcript)
        talk_time = self.conversation_analyzer.analyze_talk_time(transcript)
        questions = self.conversation_analyzer.count_questions(transcript)
        interruptions = self.conversation_analyzer.detect_interruptions(transcript)

        # Conversation context
        context_info = f"""
Conversation Context:
- Speakers: {', '.join(speakers) if speakers else 'Unknown'}
- Total Questions: {questions.get('total', 0)}
- Interruptions: {interruptions}
"""

        # Enhanced extraction hints
        extraction_hints = """
EXTRACTION GUIDELINES:
1. People: Extract full names, roles, companies, contact info, decision-making power
2. Products: Include categories, features discussed, pricing, customer interest level
3. Financial: Budget ranges, payment terms, discount requests, deal values
4. Action Items: Assign to specific people, include timelines and dependencies
5. Issues: Categorize by severity, include affected systems and workarounds
6. Competitors: Note how they're mentioned (positive/negative/comparison)
7. Key Moments: Capture objection handling, pain points, buying signals
8. Sentiment: Consider both overall and speaker-specific sentiment
9. Opportunities: Assess sales stage, probability, timeline, decision criteria

QUALITY REQUIREMENTS:
- Confidence scores should reflect actual certainty
- Use null/empty for unknown fields rather than guessing
- Include relevant context snippets for important extractions
- Flag any risks or red flags in the conversation
"""

        user_message = f"""{mode_instructions.get(mode, mode_instructions[ExtractionMode.GENERAL])}

{context_info}

{extraction_hints}

TRANSCRIPT:
{transcript}

Return comprehensive JSON matching the schema. Focus on accuracy over completeness."""

        return [
            {
                "role": "system",
                "content": f"You are an expert business intelligence analyst. Extract comprehensive structured data from conversations. Return ONLY valid JSON matching the provided schema.\n\nSCHEMA: {schema_str}"
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

    def extract(self, transcript: str, mode: Optional[ExtractionMode] = None) -> Dict[str, Any]:
        """Extract enhanced business intelligence from transcript"""
        start_time = time.time()
        self.metrics["total_extractions"] += 1

        # Determine extraction mode
        if mode is None:
            mode = self.mode
        if mode == ExtractionMode.AUTO_DETECT:
            mode = self.auto_detect_mode(transcript)

        self.metrics["mode_usage"][mode.value] += 1

        try:
            # Build enhanced prompt
            messages = self.build_enhanced_prompt(transcript, mode)

            # Call API with longer context
            response = self._call_api(messages, mode)

            # Get appropriate schema class
            schema_class = self.get_schema_class(mode)

            try:
                # Validate with mode-specific schema
                intelligence = schema_class.model_validate(response)

                # Add conversation metrics
                self._enhance_with_conversation_metrics(intelligence, transcript)

                # Success result
                result = {
                    "success": True,
                    "mode": mode.value,
                    "data": intelligence.model_dump(),
                    "error": None,
                    "latency": time.time() - start_time,
                    "conversation_stats": {
                        "speakers": self.conversation_analyzer.extract_speakers(transcript),
                        "talk_time_distribution": self.conversation_analyzer.analyze_talk_time(transcript),
                        "question_analysis": self.conversation_analyzer.count_questions(transcript),
                        "interruptions": self.conversation_analyzer.detect_interruptions(transcript)
                    }
                }

                self.metrics["successful_extractions"] += 1

            except ValidationError as e:
                # Return partial results with validation errors
                result = {
                    "success": False,
                    "mode": mode.value,
                    "data": response if isinstance(response, dict) else {},
                    "error": f"Validation error: {str(e)}",
                    "latency": time.time() - start_time
                }
                self.metrics["failed_extractions"] += 1

        except Exception as e:
            # Handle API or other errors
            result = {
                "success": False,
                "mode": mode.value if mode else "unknown",
                "data": {},
                "error": f"Extraction error: {str(e)}",
                "latency": time.time() - start_time
            }
            self.metrics["failed_extractions"] += 1

        # Update average latency
        self._update_latency_metrics(result["latency"])

        return result

    def _call_api(self, messages: List[Dict[str, str]], mode: ExtractionMode) -> Dict[str, Any]:
        """Call the vLLM API with mode-specific parameters"""

        # Adjust parameters based on mode complexity
        max_tokens = 1000 if mode == ExtractionMode.GENERAL else 1500
        temperature = 0.1 if mode in [ExtractionMode.SALES, ExtractionMode.SUPPORT] else 0.2

        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
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

    def _enhance_with_conversation_metrics(self, intelligence: BaseModel, transcript: str):
        """Enhance the intelligence object with conversation analysis"""
        if hasattr(intelligence, 'conversation_metrics'):
            speakers = self.conversation_analyzer.extract_speakers(transcript)
            talk_time = self.conversation_analyzer.analyze_talk_time(transcript)
            questions = self.conversation_analyzer.count_questions(transcript)
            interruptions = self.conversation_analyzer.detect_interruptions(transcript)

            # Update conversation metrics
            intelligence.conversation_metrics.total_speakers = len(speakers)
            intelligence.conversation_metrics.interruptions = interruptions
            intelligence.conversation_metrics.question_count = questions.get('total', 0)

            # Identify customer vs agent talk time (heuristic)
            if len(speakers) == 2:
                speaker_times = list(talk_time.values())
                if speaker_times:
                    # Assume first speaker is agent, second is customer (can be improved)
                    intelligence.conversation_metrics.agent_talk_time_percent = speaker_times[0]
                    intelligence.conversation_metrics.customer_talk_time_percent = speaker_times[1]

    def _update_latency_metrics(self, latency: float):
        """Update average latency metrics"""
        total_extractions = self.metrics["total_extractions"]
        current_avg = self.metrics["avg_latency"]

        # Update running average
        self.metrics["avg_latency"] = ((current_avg * (total_extractions - 1)) + latency) / total_extractions

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        success_rate = 0.0
        if self.metrics["total_extractions"] > 0:
            success_rate = self.metrics["successful_extractions"] / self.metrics["total_extractions"]

        return {
            **self.metrics,
            "success_rate": success_rate
        }

    def close(self):
        """Close HTTP client"""
        self.client.close()


# Example usage
if __name__ == "__main__":
    # Sample sales conversation
    sales_transcript = """
    [Sales Call Recording 2024-01-15]

    Agent Sarah: Hi John, thanks for taking the time to speak with me today about our CRM solution.

    John Miller: Of course, Sarah. I'm the VP of Sales at TechCorp Inc. We're looking to upgrade from our current system.

    Agent Sarah: Great! What challenges are you facing with your current CRM?

    John Miller: Well, our team of 25 sales reps is struggling with data entry and reporting. We're using an old system that's costing us $500 per month but it's not meeting our needs.

    Agent Sarah: I understand. Our Enterprise package at $1,200 per month for 25 users includes automated data entry and advanced analytics. Would you like to see a demo next week?

    John Miller: That sounds interesting, but the price is concerning. We budgeted around $800 monthly. Is there any flexibility on pricing?

    Agent Sarah: I can potentially offer a 15% discount for an annual commitment. That would bring it to $1,020 monthly. We also have integration with Salesforce which I know you mentioned using.

    John Miller: That's better. I need to discuss with our CFO. Can you send a proposal to john.miller@techcorp.com? Our decision timeline is end of Q1.

    Agent Sarah: Absolutely! I'll send the proposal today. Should I include the ROI calculator that shows potential savings?

    John Miller: Yes, please do. Our main criteria are cost, ease of use, and integration capabilities.
    """

    # Initialize enhanced extractor
    extractor = EnhancedExtractor(mode=ExtractionMode.SALES)

    try:
        # Extract comprehensive intelligence
        result = extractor.extract(sales_transcript)

        print("Enhanced Extraction Result:")
        print(json.dumps(result, indent=2))

        print("\nExtractor Metrics:")
        print(json.dumps(extractor.get_metrics(), indent=2))

    except Exception as e:
        print(f"Error: {e}")

    finally:
        extractor.close()