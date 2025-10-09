"""
Data models for S2A SDK responses and requests
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class JobStatusType(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class IntelligenceMode(str, Enum):
    AUTO_DETECT = "auto_detect"
    SALES = "sales"
    SUPPORT = "support"
    GENERAL = "general"
    QUICK = "quick"


class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"
    OGG = "ogg"


class Priority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class ActionItem:
    """Represents an action item extracted from conversation"""
    task: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    priority: str = "medium"
    confidence: float = 0.8


@dataclass
class Person:
    """Person entity extracted from conversation"""
    name: str
    role: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    is_decision_maker: Optional[bool] = None


@dataclass
class Product:
    """Product/service discussed in conversation"""
    name: str
    category: Optional[str] = None
    quantity: Optional[int] = None
    price: Optional[float] = None
    features_discussed: List[str] = None

    def __post_init__(self):
        if self.features_discussed is None:
            self.features_discussed = []


@dataclass
class FinancialInfo:
    """Financial information extracted"""
    amounts: List[float] = None
    budget_range: Optional[Dict[str, float]] = None
    currency: str = "USD"
    discount_requests: List[float] = None

    def __post_init__(self):
        if self.amounts is None:
            self.amounts = []
        if self.discount_requests is None:
            self.discount_requests = []


@dataclass
class ConversationMetrics:
    """Conversation quality and analysis metrics"""
    total_speakers: int = 0
    customer_talk_time_percent: Optional[float] = None
    agent_talk_time_percent: Optional[float] = None
    question_count: int = 0
    interruptions: int = 0
    pace_rating: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription"""
    job_id: str
    text: str
    duration: float
    confidence: float
    processing_time: float
    rtf: float  # Real-time factor
    status:str
    chunks: int = 1
    audio_quality: Optional[Dict[str, Any]] = None


@dataclass
class QuickIntelligenceResult:
    """Quick intelligence extraction (1-2 seconds)"""
    summary: str
    intent: str
    sentiment: str
    action_items: List[ActionItem]
    key_entities: List[str]
    confidence_score: float
    processing_time: float


@dataclass
class IntelligenceResult:
    """Comprehensive intelligence extraction (5-15 seconds)"""
    # Core classification
    call_type: str
    intent: str
    sentiment: str
    summary: str
    key_topics: List[str]

    # Extracted entities
    people: List[Person]
    companies: List[str]
    products: List[Product]
    action_items: List[ActionItem]

    # Contact information
    emails: List[str]
    phones: List[str]
    dates: List[str]

    # Financial data
    financial_info: FinancialInfo

    # Business context
    opportunity_info: Optional[Dict[str, Any]] = None
    issues: List[Dict[str, Any]] = None

    # Conversation analysis
    conversation_metrics: ConversationMetrics = None

    # Quality scores
    confidence_score: float = 0.8
    completeness_score: float = 0.8

    # AI recommendations
    recommendations: List[str] = None
    risk_flags: List[str] = None

    def __post_init__(self):
        if self.key_topics is None:
            self.key_topics = []
        if self.people is None:
            self.people = []
        if self.companies is None:
            self.companies = []
        if self.products is None:
            self.products = []
        if self.action_items is None:
            self.action_items = []
        if self.emails is None:
            self.emails = []
        if self.phones is None:
            self.phones = []
        if self.dates is None:
            self.dates = []
        if self.issues is None:
            self.issues = []
        if self.recommendations is None:
            self.recommendations = []
        if self.risk_flags is None:
            self.risk_flags = []
        if self.conversation_metrics is None:
            self.conversation_metrics = ConversationMetrics()


@dataclass
class CompleteResult:
    """Combined transcription and intelligence result"""
    transcription: TranscriptionResult
    quick_intelligence: Optional[QuickIntelligenceResult] = None
    enhanced_intelligence: Optional[IntelligenceResult] = None

    @property
    def has_intelligence(self) -> bool:
        """Check if any intelligence data is available"""
        return self.quick_intelligence is not None or self.enhanced_intelligence is not None

    @property
    def best_intelligence(self) -> Union[IntelligenceResult, QuickIntelligenceResult, None]:
        """Get the most comprehensive intelligence available"""
        if self.enhanced_intelligence:
            return self.enhanced_intelligence
        return self.quick_intelligence


@dataclass
class AsyncJob:
    """Represents an async processing job"""
    job_id: str
    status: JobStatusType

@dataclass
class JobStatusResult:
    """Detailed transcription result when job is completed"""
    job_id: str
    status: str
    text: Optional[str] = None
    duration: Optional[float] = None
    rtf: Optional[float] = None
    processing_time: Optional[float] = None
    chunks: Optional[int] = None
    confidence: Optional[float] = None
    audio_quality: Optional[Dict] = None
    # Intelligence fields
    quick_intelligence: Optional["QuickIntelligence"] = None
    enhanced_intelligence_status: Optional["EnhancedIntelligenceStatus"] = None


@dataclass
class JobStatus:
    """Current status of a transcription job (mirrors StatusResponse)"""
    job_id: str
    status: str   # pending, processing, completed, failed, rejected
    result: Optional[JobStatusResult] = None
    error: Optional[str] = None


# Exception Classes
class S2AError(Exception):
    """Base exception for S2A SDK errors"""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class AuthenticationError(S2AError):
    """Authentication or authorization failed"""
    pass


class RateLimitError(S2AError):
    """API rate limit exceeded"""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class AudioValidationError(S2AError):
    """Audio file validation failed"""
    pass


class TimeoutError(S2AError):
    """Request or processing timeout"""
    pass


class IntelligenceUnavailableError(S2AError):
    """Intelligence service is not available"""
    pass