from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any

class QuickIntelligence(BaseModel):
    """Quick intelligence results (1-2 seconds)"""
    summary: str
    intent: str
    sentiment: str
    action_items: List[Dict[str, Any]]
    key_entities: List[str]
    confidence_score: float
    processing_time: float

class EnhancedIntelligenceStatus(BaseModel):
    """Status of enhanced intelligence processing"""
    job_id: str
    status: str
    estimated_completion: Optional[str] = None

class TranscriptionResponse(BaseModel):
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
    quick_intelligence: Optional[QuickIntelligence] = None
    enhanced_intelligence_status: Optional[EnhancedIntelligenceStatus] = None

class TranscribeAsyncResponse(BaseModel):
    job_id: str
    status: str
    reason: Optional[str] = None

class StatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[TranscriptionResponse] = None
    error: Optional[str] = None
