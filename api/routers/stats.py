from typing import Any, Dict
from fastapi import APIRouter, Response, Depends, Request
from api.schemas import HealthResponse
import time
import torch
from dependencies import get_services
from db_services.auth import require_permission, get_rate_limit_headers, APIKey



router = APIRouter(prefix="/statistics", tags=["Statistics"])

@router.get("/health", response_model=HealthResponse)
async def health_check(
    request:Request,
    services = Depends(get_services)
):
    """Health check endpoint - public access for monitoring"""
    asr_svc, audio_proc, batch_proc = services
    batch_processor_stats = await batch_proc.get_queue_stats()
    return HealthResponse(
        status="healthy",
        model_info=asr_svc.get_model_info(),
        gpu_available=torch.cuda.is_available(),
        batch_processor_stats=batch_processor_stats,
        uptime=time.time() - request.app.state.app_start_time
    )

@router.get("/stats", response_model=Dict[str, Any])
async def get_service_stats(
    request: Request,
    response: Response,
    key_info: APIKey = Depends(require_permission("stats")),
    services = Depends(get_services)
):
    """Get service performance statistics - requires stats permission"""
    asr_svc, audio_proc, batch_proc = services
    
    # Add rate limit headers
    headers = get_rate_limit_headers(request)
    for key, value in headers.items():
        response.headers[key] = value
    
    return {
        "model_info": asr_svc.get_model_info(),
        "batch_processor": batch_proc.get_stats(),
        "uptime": time.time() - request.app.state.app_start_time,
        "api_key_info": {
            "key_id": key_info.key_id,
            "name": key_info.name,
            "usage_count": key_info.usage_count,
            "total_audio_minutes": key_info.total_audio_minutes
        }
    }
