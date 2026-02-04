from typing import Any, Dict
from fastapi import APIRouter, Response, Depends, Request
from api.schemas import HealthResponse
import time
import torch
from db_services.auth import require_permission, get_rate_limit_headers, APIKey



router = APIRouter(prefix="/statistics", tags=["Statistics"])

@router.get("/health", response_model=HealthResponse)
async def health_check(
    request:Request,
):
    """Health check endpoint - public access for monitoring"""
    return HealthResponse(
        status="healthy",
        uptime=time.time() - request.app.state.app_start_time
    )

@router.get("/stats", response_model=Dict[str, Any])
async def get_service_stats(
    request: Request,
    response: Response,
    key_info: APIKey = Depends(require_permission("stats")),
):

    
    # Add rate limit headers
    headers = get_rate_limit_headers(request)
    for key, value in headers.items():
        response.headers[key] = value
    
    return {
        "uptime": time.time() - request.app.state.app_start_time,
        "api_key_info": {
            "key_id": key_info.key_id,
            "name": key_info.name,
            "usage_count": key_info.usage_count,
            "total_audio_minutes": key_info.total_audio_minutes
        }
    }
