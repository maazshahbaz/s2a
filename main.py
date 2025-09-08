from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import tempfile
import os
import uuid
import asyncio
from pathlib import Path
from loguru import logger
import time
from contextlib import asynccontextmanager

import torch
from asr_service import NeMoASRService, TranscriptionResult
from audio_utils import AudioProcessor
from batch_processor import BatchProcessor, BatchConfig, BatchJob
from config import ASRConfig, get_settings
from performance_monitor import get_performance_monitor
from auth import auth, require_permission, update_usage, get_rate_limit_headers, APIKey
from webhook import webhook_sender, WebhookPayload

class TranscriptionRequest(BaseModel):
    audio_file: str = Field(description="Path to audio file or file upload")
    enhance_audio: bool = Field(default=True, description="Apply audio enhancement")
    remove_silence: bool = Field(default=False, description="Remove silence from audio")
    priority: int = Field(default=0, description="Processing priority (higher = more priority)")
    callback_url: Optional[str] = Field(default=None, description="Webhook URL for async results")

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

class HealthResponse(BaseModel):
    status: str
    model_info: Dict[str, Any]
    gpu_available: bool
    batch_processor_stats: Dict[str, Any]
    uptime: float

class StatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[TranscriptionResponse] = None
    error: Optional[str] = None

# Global variables for services
asr_service = None
audio_processor = None
batch_processor = None
app_start_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global asr_service, audio_processor, batch_processor, app_start_time
    
    settings = get_settings()
    app_start_time = time.time()
    
    logger.info("Initializing ASR microservice...")
    
    # Initialize services
    asr_service = NeMoASRService(
        model_name=settings.model_name,
        device=settings.device,
        batch_size=settings.batch_size,
        max_chunk_duration=settings.max_chunk_duration,
        min_audio_duration=settings.min_audio_duration
    )
    
    audio_processor = AudioProcessor(
        target_sr=settings.target_sample_rate,
        vad_aggressiveness=settings.vad_aggressiveness
    )
    
    batch_config = BatchConfig(
        max_batch_size=settings.batch_size,
        max_queue_size=settings.max_queue_size,
        processing_timeout=settings.processing_timeout,
        dynamic_batching=settings.dynamic_batching,
        batch_timeout_ms=settings.batch_timeout_ms,
        gpu_memory_fraction=settings.gpu_memory_fraction
    )
    
    batch_processor = BatchProcessor(
        asr_service=asr_service,
        config=batch_config,
        num_workers=settings.num_workers
    )
    
    # Start batch processor
    await batch_processor.start()
    
    logger.info("ASR microservice initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ASR microservice...")
    
    if batch_processor:
        await batch_processor.stop()
    
    logger.info("ASR microservice shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="S2A Speech-to-Text Microservice",
    description="High-performance ASR service using NVIDIA NeMo Parakeet model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development
        "http://localhost:8080",  # Vue development
        "https://your-domain.com"  # Production domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Dependency to get services
def get_services():
    if not all([asr_service, audio_processor, batch_processor]):
        raise HTTPException(status_code=503, detail="Services not initialized")
    return asr_service, audio_processor, batch_processor

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "BytePulse AI S2A Speech-to-Text Microservice", 
        "status": "running",
        "version": "1.0.0",
        "authentication": "required"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(
    services = Depends(get_services)
):
    """Health check endpoint - public access for monitoring"""
    asr_svc, audio_proc, batch_proc = services
    
    return HealthResponse(
        status="healthy",
        model_info=asr_svc.get_model_info(),
        gpu_available=torch.cuda.is_available(),
        batch_processor_stats=batch_proc.get_stats(),
        uptime=time.time() - app_start_time
    )

@app.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe_sync(
    request: Request,
    response: Response,
    audio_file: UploadFile = File(...),
    enhance_audio: bool = True,
    remove_silence: bool = False,
    key_info: APIKey = Depends(require_permission("transcribe")),
    services = Depends(get_services)
):
    """Synchronous transcription endpoint - requires transcribe permission"""
    asr_svc, audio_proc, batch_proc = services
    job_id = str(uuid.uuid4())
    
    # Add rate limit headers
    headers = get_rate_limit_headers(request)
    for key, value in headers.items():
        response.headers[key] = value
    print(audio_file)
    if not audio_file.content_type.startswith(('audio/', 'video/')):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be audio or video.")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_file:
        content = await audio_file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Process audio
        audio, sr, audio_info = audio_proc.process_audio_file(
            tmp_file_path,
            enhance=enhance_audio,
            validate=True
        )
        
        # Check minimum duration (both sync/async same rule)
        if audio_info['duration'] < asr_svc.min_audio_duration:
            return TranscriptionResponse(
                job_id=job_id,
                status="rejected",
                text="Audio too short. Minimum duration is 5 seconds.",
                duration=audio_info['duration'],
                rtf=0,
                processing_time=0
            )
        
        # Check maximum duration for SYNC API (2 minutes max)
        settings = get_settings()
        if audio_info['duration'] > settings.max_sync_audio_duration:
            return TranscriptionResponse(
                job_id=job_id,
                status="rejected", 
                text=f"Audio too long for sync API. Maximum duration is {settings.max_sync_audio_duration/60:.0f} minutes. Please use async API for longer audio.",
                duration=audio_info['duration'],
                rtf=0,
                processing_time=0
            )
        
        # Transcribe using ASR service
        result = await asr_svc.transcribe_audio(tmp_file_path)
        
        # Update usage statistics
        update_usage(request, result.duration)
        
        return TranscriptionResponse(
            job_id=job_id,
            status="completed",
            text=result.text,
            duration=result.duration,
            rtf=result.rtf,
            processing_time=result.processing_time,
            chunks=len(result.chunks) if result.chunks else 1,
            confidence=result.confidence,
            audio_quality=audio_info.get('quality_metrics')
        )
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.post("/v1/transcribe/async", response_model=Dict[str, str])
async def transcribe_async(
    request: Request,
    response: Response,
    callback_url: str = Form(...),
    audio_file: UploadFile = File(...),
    enhance_audio: bool = True,
    remove_silence: bool = False,
    priority: int = 0,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    key_info: APIKey = Depends(require_permission("transcribe")),
    services = Depends(get_services)
):
    """Asynchronous transcription endpoint - requires transcribe permission and callback_url"""
    asr_svc, audio_proc, batch_proc = services
    job_id = str(uuid.uuid4())
    
    # Validate callback URL
    if not webhook_sender.validate_callback_url(callback_url):
        raise HTTPException(status_code=400, detail="Invalid callback_url. Must be a valid HTTP/HTTPS URL.")
    
    # Add rate limit headers
    headers = get_rate_limit_headers(request)
    for key, value in headers.items():
        response.headers[key] = value
    
    if not audio_file.content_type.startswith(('audio/', 'video/')):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be audio or video.")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_file:
        content = await audio_file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    # Add to background processing
    background_tasks.add_task(
        process_audio_background,
        job_id, tmp_file_path, enhance_audio, remove_silence, priority, callback_url,
        asr_svc, audio_proc, batch_proc
    )
    
    return {"job_id": job_id, "status": "accepted"}

async def process_audio_background(
    job_id: str,
    audio_path: str,
    enhance_audio: bool,
    remove_silence: bool,
    priority: int,
    callback_url: str,
    asr_svc,
    audio_proc,
    batch_proc
):
    """Background task for async audio processing"""
    try:
        # Process audio
        audio, sr, audio_info = audio_proc.process_audio_file(
            audio_path,
            enhance=enhance_audio,
            validate=True
        )
        
        # Check minimum duration (both sync/async same rule)
        if audio_info['duration'] < asr_svc.min_audio_duration:
            logger.info(f"Job {job_id}: Audio too short ({audio_info['duration']:.1f}s < {asr_svc.min_audio_duration:.1f}s), skipping")
            return
        
        # Check maximum duration for ASYNC API (2 hours max)
        from config import get_settings
        settings = get_settings()
        if audio_info['duration'] > settings.max_async_audio_duration:
            logger.info(f"Job {job_id}: Audio too long ({audio_info['duration']:.1f}s > {settings.max_async_audio_duration:.1f}s), skipping")
            return
        
        # Submit to batch processor
        result = await batch_proc.transcribe_async(
            job_id=job_id,
            audio_data=audio,
            metadata={
                'sample_rate': sr,
                'duration': audio_info['duration'],
                'quality_metrics': audio_info.get('quality_metrics'),
                'enhancement_applied': enhance_audio,
                'callback_url': callback_url
            },
            priority=priority,
            timeout=300.0
        )
        
        # Send webhook with results
        if result:
            webhook_payload = WebhookPayload(
                job_id=job_id,
                status="completed",
                result=result,
                processing_time=result.get('processing_time')
            )
        else:
            webhook_payload = WebhookPayload(
                job_id=job_id,
                status="failed",
                error="Transcription processing failed"
            )
        
        # Send webhook asynchronously (don't wait for it)
        asyncio.create_task(webhook_sender.send_webhook(callback_url, webhook_payload))
        
        logger.info(f"Job {job_id} completed and webhook sent to {callback_url}")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        
        # Send error webhook
        error_payload = WebhookPayload(
            job_id=job_id,
            status="failed",
            error=str(e)
        )
        asyncio.create_task(webhook_sender.send_webhook(callback_url, error_payload))
    
    finally:
        # Clean up temporary file
        if os.path.exists(audio_path):
            os.unlink(audio_path)

@app.get("/v1/status/{job_id}", response_model=StatusResponse)
async def get_transcription_status(
    job_id: str,
    request: Request,
    response: Response,
    key_info: APIKey = Depends(require_permission("status")),
    services = Depends(get_services)
):
    """Get status of async transcription job - requires status permission"""
    asr_svc, audio_proc, batch_proc = services
    
    # Add rate limit headers
    headers = get_rate_limit_headers(request)
    for key, value in headers.items():
        response.headers[key] = value
    
    # Try to get result from batch processor
    result_job = await batch_proc.batcher.get_result(job_id, timeout=0.1)
    
    if result_job is None:
        # Check if job is still processing
        async with batch_proc.batcher._lock:
            if job_id in batch_proc.batcher.processing_jobs:
                return StatusResponse(job_id=job_id, status="processing")
        
        return StatusResponse(job_id=job_id, status="not_found")
    
    if result_job.status.value == "failed":
        return StatusResponse(
            job_id=job_id,
            status="failed",
            error=result_job.error
        )
    
    # Convert result to response format
    result = result_job.result
    response = TranscriptionResponse(
        job_id=job_id,
        status="completed",
        text=result.get('text', ''),
        duration=result.get('duration', 0),
        rtf=result.get('rtf', 0),
        processing_time=result.get('processing_time', 0),
        chunks=1,  # Batch processing uses single chunks
        confidence=None,
        audio_quality=result_job.metadata.get('quality_metrics')
    )
    
    return StatusResponse(
        job_id=job_id,
        status="completed",
        result=response
    )

@app.get("/v1/stats", response_model=Dict[str, Any])
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
        "uptime": time.time() - app_start_time,
        "api_key_info": {
            "key_id": key_info.key_id,
            "name": key_info.name,
            "usage_count": key_info.usage_count,
            "total_audio_minutes": key_info.total_audio_minutes
        }
    }

@app.delete("/v1/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    request: Request,
    response: Response,
    key_info: APIKey = Depends(require_permission("transcribe")),
    services = Depends(get_services)
):
    """Cancel a pending or processing job - requires transcribe permission"""
    asr_svc, audio_proc, batch_proc = services
    
    # Add rate limit headers
    headers = get_rate_limit_headers(request)
    for key, value in headers.items():
        response.headers[key] = value
    
    # Note: This is a simplified implementation
    # In production, you'd want more sophisticated job cancellation
    async with batch_proc.batcher._lock:
        if job_id in batch_proc.batcher.processing_jobs:
            job = batch_proc.batcher.processing_jobs[job_id]
            job.status = "cancelled"
            return {"message": f"Job {job_id} cancelled"}
    
    return {"message": f"Job {job_id} not found or cannot be cancelled"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1,  # Single worker due to GPU memory constraints
        log_level="info"
    )