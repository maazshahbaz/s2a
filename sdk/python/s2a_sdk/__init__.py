"""
S2A Python SDK - Speech-to-Actions with Business Intelligence

A comprehensive SDK for the S2A speech-to-text and intelligence platform.
Provides easy-to-use interfaces for transcription, intelligence extraction,
and combined workflows.

Example:
    ```python
    from s2a_sdk import S2AClient

    client = S2AClient(api_key="bp-proj-your-key")

    # Simple transcription
    result = client.transcribe("audio.wav")

    # Transcription with intelligence
    result = client.transcribe_with_intelligence("meeting.mp3")
    print(f"Summary: {result.intelligence.summary}")
    print(f"Action Items: {len(result.intelligence.action_items)}")
    ```
"""

from .client import S2AClient
from .models import (
    TranscriptionResult,
    IntelligenceResult,
    QuickIntelligenceResult,
    CompleteResult,
    AsyncJob,
    JobStatus,
    S2AError,
    AudioValidationError,
    RateLimitError,
    AuthenticationError
)
from .enums import (
    JobStatusType,
    IntelligenceMode,
    AudioFormat,
    Priority
)

__version__ = "1.0.0"
__author__ = "99Technologies AI"
__description__ = "Official Python SDK for S2A Speech-to-Actions Platform"

__all__ = [
    # Core client
    "S2AClient",

    # Result models
    "TranscriptionResult",
    "IntelligenceResult",
    "QuickIntelligenceResult",
    "CompleteResult",
    "AsyncJob",
    "JobStatus",

    # Exceptions
    "S2AError",
    "AudioValidationError",
    "RateLimitError",
    "AuthenticationError",

    # Enums
    "JobStatusType",
    "IntelligenceMode",
    "AudioFormat",
    "Priority"
]