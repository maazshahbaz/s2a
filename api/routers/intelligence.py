#!/usr/bin/env python3
"""
Intelligence API endpoints for S2A Pipeline
Provides access to enhanced business intelligence extraction capabilities
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from intelligence.intelligence_service import get_intelligence_service
from intelligence.enhanced_extractor import ExtractionMode
from dependencies import get_api_key


router = APIRouter(
    prefix="/intelligence",
    tags=["intelligence"],
    responses={
        401: {"description": "Authentication required"},
        500: {"description": "Internal server error"}
    }
)


# Request/Response Models
class IntelligenceRequest(BaseModel):
    transcript_id: str = Field(..., description="Unique identifier for the transcript")
    transcript_text: str = Field(..., description="The transcription text to analyze")
    mode: Optional[ExtractionMode] = Field(ExtractionMode.AUTO_DETECT, description="Extraction mode")
    priority: Optional[str] = Field("normal", description="Processing priority: high, normal, low")


class IntelligenceJobResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    transcript_id: str = Field(..., description="Source transcript ID")
    status: str = Field(..., description="Current job status")
    message: str = Field(..., description="Human-readable status message")


class IntelligenceResultResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier")
    transcript_id: str = Field(..., description="Source transcript ID")
    status: str = Field(..., description="Job status")
    mode: str = Field(..., description="Extraction mode used")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    intelligence: Optional[Dict[str, Any]] = Field(None, description="Extracted intelligence data")
    conversation_stats: Optional[Dict[str, Any]] = Field(None, description="Conversation analysis metrics")
    error: Optional[str] = Field(None, description="Error message if failed")


class IntelligenceMetricsResponse(BaseModel):
    service_metrics: Dict[str, Any] = Field(..., description="Intelligence service metrics")
    queue_status: Dict[str, Any] = Field(..., description="Current queue status")
    health: Dict[str, Any] = Field(..., description="Service health information")


@router.post("/extract", response_model=IntelligenceJobResponse)
async def extract_intelligence(
    request: IntelligenceRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
) -> IntelligenceJobResponse:
    """
    Submit a transcript for intelligence extraction

    This endpoint accepts a transcript and submits it for comprehensive business intelligence
    extraction including action items, entities, sentiment analysis, and more.
    """
    try:
        intelligence_service = await get_intelligence_service()

        # Submit job for processing
        job_id = await intelligence_service.submit_job(
            transcript_id=request.transcript_id,
            transcript_text=request.transcript_text,
            mode=request.mode,
            priority=request.priority
        )

        return IntelligenceJobResponse(
            job_id=job_id,
            transcript_id=request.transcript_id,
            status="submitted",
            message="Intelligence extraction job submitted successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit intelligence job: {str(e)}"
        )


@router.get("/job/{job_id}/status", response_model=Dict[str, Any])
async def get_job_status(
    job_id: str,
    api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Get the status of an intelligence extraction job

    Returns the current status and progress of a specific intelligence job.
    """
    try:
        intelligence_service = await get_intelligence_service()
        status = await intelligence_service.get_job_status(job_id)

        if status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )

        return status

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/job/{job_id}/result", response_model=IntelligenceResultResponse)
async def get_job_result(
    job_id: str,
    api_key: str = Depends(get_api_key)
) -> IntelligenceResultResponse:
    """
    Get the result of a completed intelligence extraction job

    Returns the complete intelligence data extracted from the transcript.
    """
    try:
        intelligence_service = await get_intelligence_service()

        # Get job status first
        job_status = await intelligence_service.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )

        # Get result if completed
        result = await intelligence_service.get_job_result(job_id)

        response = IntelligenceResultResponse(
            job_id=job_id,
            transcript_id=job_status["transcript_id"],
            status=job_status["status"],
            mode=job_status["mode"],
            processing_time=job_status.get("processing_time"),
            error=job_status.get("error")
        )

        if result and result.get("success"):
            response.intelligence = result["data"]
            response.conversation_stats = result.get("conversation_stats")

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job result: {str(e)}"
        )


@router.post("/extract/sync", response_model=IntelligenceResultResponse)
async def extract_intelligence_sync(
    request: IntelligenceRequest,
    api_key: str = Depends(get_api_key)
) -> IntelligenceResultResponse:
    """
    Synchronously extract intelligence from a transcript

    This endpoint processes the transcript immediately and returns the result.
    Use for small transcripts or when immediate results are needed.
    """
    try:
        intelligence_service = await get_intelligence_service()

        # Submit high-priority job
        job_id = await intelligence_service.submit_job(
            transcript_id=request.transcript_id,
            transcript_text=request.transcript_text,
            mode=request.mode,
            priority="high"
        )

        # Wait for completion (with timeout)
        import asyncio
        max_wait_time = 60  # 60 seconds timeout
        check_interval = 1  # Check every second

        for _ in range(max_wait_time):
            job_status = await intelligence_service.get_job_status(job_id)

            if job_status["status"] in ["completed", "failed"]:
                # Get the result
                result = await intelligence_service.get_job_result(job_id)

                response = IntelligenceResultResponse(
                    job_id=job_id,
                    transcript_id=request.transcript_id,
                    status=job_status["status"],
                    mode=job_status["mode"],
                    processing_time=job_status.get("processing_time"),
                    error=job_status.get("error")
                )

                if result and result.get("success"):
                    response.intelligence = result["data"]
                    response.conversation_stats = result.get("conversation_stats")

                return response

            await asyncio.sleep(check_interval)

        # Timeout
        raise HTTPException(
            status_code=408,
            detail=f"Intelligence extraction timed out after {max_wait_time} seconds"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract intelligence: {str(e)}"
        )


@router.get("/metrics", response_model=IntelligenceMetricsResponse)
async def get_intelligence_metrics(
    api_key: str = Depends(get_api_key)
) -> IntelligenceMetricsResponse:
    """
    Get intelligence service metrics and status

    Returns performance metrics, queue status, and health information.
    """
    try:
        intelligence_service = await get_intelligence_service()

        metrics = await intelligence_service.get_metrics()
        queue_status = await intelligence_service.get_queue_status()
        health = await intelligence_service.health_check()

        return IntelligenceMetricsResponse(
            service_metrics=metrics,
            queue_status=queue_status,
            health=health
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get intelligence metrics: {str(e)}"
        )


@router.get("/health")
async def intelligence_health_check() -> Dict[str, Any]:
    """
    Health check endpoint for intelligence service

    Returns basic health status without requiring authentication.
    """
    try:
        intelligence_service = await get_intelligence_service()
        health = await intelligence_service.health_check()

        return {
            "service": "intelligence",
            "status": health["status"],
            "timestamp": health.get("timestamp", "unknown")
        }

    except Exception as e:
        return {
            "service": "intelligence",
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/modes")
async def get_extraction_modes() -> Dict[str, Any]:
    """
    Get available extraction modes and their descriptions

    Returns information about different intelligence extraction modes.
    """
    return {
        "modes": {
            "auto_detect": {
                "name": "Auto Detect",
                "description": "Automatically determines the best extraction mode based on transcript content"
            },
            "sales": {
                "name": "Sales",
                "description": "Optimized for sales conversations with lead qualification, pricing, and opportunity tracking"
            },
            "support": {
                "name": "Customer Support",
                "description": "Optimized for customer support with issue tracking, resolution, and satisfaction analysis"
            },
            "general": {
                "name": "General",
                "description": "General purpose extraction for meetings and business conversations"
            }
        },
        "capabilities": [
            "Action items extraction",
            "Entity recognition (people, companies, products, contacts)",
            "Financial information extraction",
            "Sentiment analysis",
            "Intent classification",
            "Conversation metrics",
            "Key moments identification",
            "Risk assessment",
            "Recommendations generation"
        ]
    }