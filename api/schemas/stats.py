from pydantic import BaseModel
from typing import  Dict, Any

class HealthResponse(BaseModel):
    status: str
    model_info: Dict[str, Any]
    gpu_available: bool
    batch_processor_stats: Dict[str, Any]
    uptime: float
    