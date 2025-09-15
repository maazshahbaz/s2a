from pydantic import BaseModel, Field
from typing import  Dict, Optional

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

class TranscribeAsyncResponse(BaseModel):
    job_id: str
    status: str

class StatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[TranscriptionResponse] = None
    error: Optional[str] = None
