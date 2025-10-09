#!/usr/bin/env python3
"""
Enhanced Business Intelligence Schema for S2A Pipeline
Comprehensive extraction for sales, customer support, and general business contexts
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


# Enhanced Enums
class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Sentiment(str, Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class CallType(str, Enum):
    SALES_CALL = "sales_call"
    CUSTOMER_SUPPORT = "customer_support"
    INTERNAL_MEETING = "internal_meeting"
    TRAINING_SESSION = "training_session"
    PROJECT_REVIEW = "project_review"
    CLIENT_ONBOARDING = "client_onboarding"
    FOLLOW_UP = "follow_up"
    DEMO_PRESENTATION = "demo_presentation"
    NEGOTIATION = "negotiation"
    COMPLAINT_RESOLUTION = "complaint_resolution"

class Intent(str, Enum):
    # Sales intents
    LEAD_QUALIFICATION = "lead_qualification"
    PRODUCT_DEMO = "product_demo"
    PRICING_DISCUSSION = "pricing_discussion"
    CONTRACT_NEGOTIATION = "contract_negotiation"
    UPSELL_CROSSSELL = "upsell_crosssell"
    RENEWAL_DISCUSSION = "renewal_discussion"

    # Support intents
    TECHNICAL_SUPPORT = "technical_support"
    BILLING_INQUIRY = "billing_inquiry"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    ACCOUNT_MANAGEMENT = "account_management"
    TRAINING_REQUEST = "training_request"

    # Business intents
    PROCUREMENT = "procurement"
    VENDOR_MANAGEMENT = "vendor_management"
    PROJECT_PLANNING = "project_planning"
    STATUS_UPDATE = "status_update"
    GENERAL_INQUIRY = "general_inquiry"

class CustomerStage(str, Enum):
    PROSPECT = "prospect"
    QUALIFIED_LEAD = "qualified_lead"
    OPPORTUNITY = "opportunity"
    CUSTOMER = "customer"
    CHURNED = "churned"
    RENEWAL = "renewal"

class IssueStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class CompetitorMention(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    COMPARISON = "comparison"


# Enhanced Entity Models
class Person(BaseModel):
    name: str = Field(..., description="Full name of the person")
    role: Optional[str] = Field(None, description="Job title or role")
    company: Optional[str] = Field(None, description="Company name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    is_decision_maker: Optional[bool] = Field(None, description="Whether they make purchasing decisions")
    sentiment_towards: Optional[Sentiment] = Field(None, description="Their sentiment in the conversation")

class Product(BaseModel):
    name: str = Field(..., description="Product or service name")
    category: Optional[str] = Field(None, description="Product category")
    quantity: Optional[int] = Field(None, description="Quantity mentioned")
    price: Optional[float] = Field(None, description="Price mentioned")
    discount: Optional[float] = Field(None, description="Discount percentage")
    features_discussed: List[str] = Field(default_factory=list, description="Specific features mentioned")
    customer_interest_level: Optional[str] = Field(None, description="high/medium/low customer interest")

class ActionItem(BaseModel):
    assignee: Optional[str] = Field(None, description="Person assigned to the task")
    task: str = Field(..., description="Description of the action item")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format")
    priority: Priority = Field(Priority.MEDIUM, description="Task priority level")
    category: Optional[str] = Field(None, description="followup/demo/proposal/technical/admin")
    estimated_hours: Optional[float] = Field(None, description="Estimated time to complete")
    dependencies: List[str] = Field(default_factory=list, description="What this task depends on")
    confidence: float = Field(0.8, description="Confidence score 0-1", ge=0, le=1)
    context: Optional[str] = Field(None, description="Relevant context from transcript")

class FinancialInfo(BaseModel):
    amounts: List[float] = Field(default_factory=list, description="Monetary amounts mentioned")
    budget_range: Optional[Dict[str, float]] = Field(None, description="Budget min/max if discussed")
    payment_terms: Optional[str] = Field(None, description="Payment terms discussed")
    billing_frequency: Optional[str] = Field(None, description="monthly/quarterly/annual")
    currency: str = Field(default="USD", description="Currency mentioned")
    discount_requests: List[float] = Field(default_factory=list, description="Discount percentages requested")

class CompetitorInfo(BaseModel):
    name: str = Field(..., description="Competitor name")
    mention_type: CompetitorMention = Field(..., description="How they were mentioned")
    context: Optional[str] = Field(None, description="Context of the mention")
    comparison_points: List[str] = Field(default_factory=list, description="What was compared")

class Issue(BaseModel):
    description: str = Field(..., description="Issue description")
    severity: Priority = Field(..., description="Issue severity level")
    category: Optional[str] = Field(None, description="technical/billing/feature/bug/other")
    status: IssueStatus = Field(IssueStatus.OPEN, description="Current status")
    affected_systems: List[str] = Field(default_factory=list, description="Systems affected")
    workaround: Optional[str] = Field(None, description="Temporary workaround if provided")

class OpportunityInfo(BaseModel):
    stage: CustomerStage = Field(..., description="Current sales stage")
    value_estimate: Optional[float] = Field(None, description="Estimated deal value")
    close_probability: Optional[float] = Field(None, description="Probability of closing (0-1)")
    timeline: Optional[str] = Field(None, description="Expected timeline to close")
    next_steps: List[str] = Field(default_factory=list, description="Agreed next steps")
    decision_criteria: List[str] = Field(default_factory=list, description="Customer decision factors")
    objections: List[str] = Field(default_factory=list, description="Customer objections raised")

class ConversationMetrics(BaseModel):
    total_speakers: int = Field(default=0, description="Number of different speakers")
    customer_talk_time_percent: Optional[float] = Field(None, description="% of time customer spoke")
    agent_talk_time_percent: Optional[float] = Field(None, description="% of time agent spoke")
    interruptions: int = Field(default=0, description="Number of interruptions")
    question_count: int = Field(default=0, description="Number of questions asked")
    customer_questions: int = Field(default=0, description="Questions from customer")
    agent_questions: int = Field(default=0, description="Questions from agent")
    pace_rating: Optional[str] = Field(None, description="slow/normal/fast conversation pace")

class KeyMoments(BaseModel):
    objection_handling: List[str] = Field(default_factory=list, description="Customer objections and responses")
    pain_points: List[str] = Field(default_factory=list, description="Customer pain points identified")
    buying_signals: List[str] = Field(default_factory=list, description="Positive buying indicators")
    escalation_triggers: List[str] = Field(default_factory=list, description="What caused escalations")
    decision_points: List[str] = Field(default_factory=list, description="Key decision moments")
    breakthrough_moments: List[str] = Field(default_factory=list, description="Moments of progress/resolution")

class Entities(BaseModel):
    # Business entities
    people: List[Person] = Field(default_factory=list, description="People mentioned")
    companies: List[str] = Field(default_factory=list, description="Company names")
    products: List[Product] = Field(default_factory=list, description="Products/services discussed")
    competitors: List[CompetitorInfo] = Field(default_factory=list, description="Competitors mentioned")

    # Document/Reference entities
    invoice_ids: List[str] = Field(default_factory=list, description="Invoice identifiers")
    order_ids: List[str] = Field(default_factory=list, description="Order identifiers")
    ticket_ids: List[str] = Field(default_factory=list, description="Support ticket IDs")
    contract_numbers: List[str] = Field(default_factory=list, description="Contract references")
    case_numbers: List[str] = Field(default_factory=list, description="Case reference numbers")

    # Contact information
    emails: List[str] = Field(default_factory=list, description="Email addresses")
    phones: List[str] = Field(default_factory=list, description="Phone numbers")
    websites: List[str] = Field(default_factory=list, description="Website URLs mentioned")

    # Temporal entities
    dates: List[str] = Field(default_factory=list, description="Dates in YYYY-MM-DD format")
    meeting_times: List[str] = Field(default_factory=list, description="Scheduled meeting times")
    deadlines: List[str] = Field(default_factory=list, description="Important deadlines")

    # Financial entities
    financial_info: FinancialInfo = Field(default_factory=FinancialInfo, description="Financial information")

    # Technical entities
    software_versions: List[str] = Field(default_factory=list, description="Software versions mentioned")
    error_codes: List[str] = Field(default_factory=list, description="Error codes mentioned")
    urls_mentioned: List[str] = Field(default_factory=list, description="URLs referenced")

    # Location entities
    locations: List[str] = Field(default_factory=list, description="Geographical locations")
    time_zones: List[str] = Field(default_factory=list, description="Time zones mentioned")


class EnhancedBusinessIntelligence(BaseModel):
    """Comprehensive business intelligence extraction schema"""

    # Core classification
    call_type: CallType = Field(CallType.INTERNAL_MEETING, description="Type of conversation")
    intent: Intent = Field(Intent.GENERAL_INQUIRY, description="Primary intent classification")
    sentiment: Sentiment = Field(Sentiment.NEUTRAL, description="Overall sentiment")

    # Content analysis
    summary: str = Field(..., description="Comprehensive summary of the conversation")
    key_topics: List[str] = Field(default_factory=list, description="Main topics discussed")

    # Action tracking
    action_items: List[ActionItem] = Field(default_factory=list, description="Extracted action items")
    follow_ups: List[str] = Field(default_factory=list, description="Follow-up activities needed")

    # Entity extraction
    entities: Entities = Field(default_factory=Entities, description="All extracted entities")

    # Business context
    opportunity_info: Optional[OpportunityInfo] = Field(None, description="Sales opportunity information")
    issues: List[Issue] = Field(default_factory=list, description="Issues or problems discussed")
    key_moments: KeyMoments = Field(default_factory=KeyMoments, description="Important conversation moments")

    # Conversation analysis
    conversation_metrics: ConversationMetrics = Field(default_factory=ConversationMetrics, description="Conversation quality metrics")

    # Quality scores
    confidence_score: float = Field(0.8, description="Overall extraction confidence", ge=0, le=1)
    completeness_score: float = Field(0.8, description="How complete the extraction is", ge=0, le=1)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="AI recommendations based on conversation")
    risk_flags: List[str] = Field(default_factory=list, description="Potential risks identified")
    success_indicators: List[str] = Field(default_factory=list, description="Positive indicators")

    # Metadata
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When extraction was performed")
    model_version: str = Field(default="enhanced-v1.0", description="Schema version used")

# Specialized schemas for different use cases
class SalesIntelligence(EnhancedBusinessIntelligence):
    """Sales-focused intelligence extraction"""

    # Sales-specific fields
    lead_quality_score: Optional[float] = Field(None, description="Lead quality score 0-1", ge=0, le=1)
    sales_stage_progression: Optional[str] = Field(None, description="How the sales stage changed")
    objection_count: int = Field(default=0, description="Number of objections raised")
    commitment_level: Optional[str] = Field(None, description="low/medium/high customer commitment")
    decision_timeline: Optional[str] = Field(None, description="When customer expects to decide")
    budget_qualification: Optional[str] = Field(None, description="qualified/unqualified/unknown budget status")


class SupportIntelligence(EnhancedBusinessIntelligence):
    """Customer support-focused intelligence extraction"""

    # Support-specific fields
    customer_satisfaction: Optional[Sentiment] = Field(None, description="Customer satisfaction level")
    resolution_time_expectation: Optional[str] = Field(None, description="Expected resolution time")
    escalation_risk: Optional[str] = Field(None, description="low/medium/high escalation risk")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Areas where agent needed help")
    customer_effort_score: Optional[int] = Field(None, description="Estimated customer effort 1-10", ge=1, le=10)
    first_call_resolution: Optional[bool] = Field(None, description="Was issue resolved in first call")

class IntelligenceMetrics(BaseModel):
    """Intelligence service metrics"""
    total_jobs_processed: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    average_processing_time: float = 0.0
    queue_size: int = 0
    active_workers: int = 0
    uptime_hours: float = 0.0
    last_job_processed: Optional[str] = None

    # Mode-specific metrics
    sales_jobs: int = 0
    support_jobs: int = 0
    general_jobs: int = 0

    # Quality metrics
    avg_confidence_score: float = 0.0
    extraction_field_rates: Dict[str, float] = {}