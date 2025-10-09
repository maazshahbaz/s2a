from typing import Dict, Any, Optional
from dataclasses import dataclass
from .enhanced_extractor import  ExtractionMode
from enum import Enum
from datetime import datetime

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class IntelligenceJob:
    """Intelligence processing job"""
    job_id: str
    transcript_id: str
    transcript_text: str
    mode: ExtractionMode = ExtractionMode.AUTO_DETECT
    priority: str = "normal"  # high, normal, low
    created_at: datetime = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()