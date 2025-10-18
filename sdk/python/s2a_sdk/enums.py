"""
Enums and constants for S2A SDK
"""

from enum import Enum


class JobStatusType(str, Enum):
    """Status types for async jobs"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class IntelligenceMode(str, Enum):
    """Intelligence extraction modes"""
    AUTO_DETECT = "auto_detect"
    SALES = "sales"
    SUPPORT = "support"
    GENERAL = "general"
    QUICK = "quick"


class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"
    OGG = "ogg"


class Priority(str, Enum):
    """Processing priority levels"""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class CallType(str, Enum):
    """Types of calls/conversations"""
    SALES_CALL = "sales_call"
    CUSTOMER_SUPPORT = "customer_support"
    INTERNAL_MEETING = "internal_meeting"
    TRAINING_SESSION = "training_session"
    PROJECT_REVIEW = "project_review"
    CLIENT_ONBOARDING = "client_onboarding"
    FOLLOW_UP = "follow_up"
    DEMO_PRESENTATION = "demo_presentation"


class Intent(str, Enum):
    """Conversation intents"""
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

    # General intents
    MEETING_FOLLOW_UP = "meeting_follow_up"
    PROJECT_UPDATE = "project_update"
    GENERAL_DISCUSSION = "general_discussion"


class Sentiment(str, Enum):
    """Sentiment classifications"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


# API Constants
DEFAULT_BASE_URL = "https://api.bytepulseai.com"
DEFAULT_TIMEOUT = 300  # 5 minutes
MIN_ASYNC_AUDIO_DURATION = 1  # 1 second
MAX_ASYNC_AUDIO_DURATION = 18000  # 5 hours in seconds

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_RETRY_BACKOFF = 2.0

# File size limits
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
MIN_AUDIO_DURATION = 5.0  # 5 seconds

# Supported MIME types
SUPPORTED_MIME_TYPES = {
    "audio/wav": AudioFormat.WAV,
    "audio/wave": AudioFormat.WAV,
    "audio/mpeg": AudioFormat.MP3,
    "audio/mp3": AudioFormat.MP3,
    "audio/flac": AudioFormat.FLAC,
    "audio/m4a": AudioFormat.M4A,
    "audio/mp4": AudioFormat.M4A,
    "audio/ogg": AudioFormat.OGG,
    "video/mp4": AudioFormat.M4A,  # Video files with audio track
}