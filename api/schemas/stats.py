from pydantic import BaseModel
from typing import  Dict, Any

class HealthResponse(BaseModel):
    status: str
    uptime: float
    