from fastapi import APIRouter, Response, UploadFile, File, Depends, Request, HTTPException, Form, BackgroundTasks
from api.schemas import  TranscriptionResponse, TranscribeAsyncResponse, StatusResponse
import os
from dependencies import get_services
from db_services.auth import require_permission, update_usage, update_request_usage, get_rate_limit_headers, APIKey
import uuid
import tempfile
from config import get_settings
from pathlib import Path
from loguru import logger
from webhook import webhook_sender
from dependencies import process_audio_background

router = APIRouter(prefix="/transcription", tags=["Transcription"])

@router.post("/transcribe", response_model=TranscriptionResponse)
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
        await update_usage(request, result.duration)
        
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


@router.post("/transcribe/async", response_model=TranscribeAsyncResponse)
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
    
    # Track API request usage immediately
    await update_request_usage(request)
    
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
        asr_svc, audio_proc, batch_proc, request.state.api_key
    )
    
    return TranscribeAsyncResponse(job_id=job_id, status="accepted")

@router.get("/status/{job_id}", response_model=StatusResponse)
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


@router.delete("/jobs/{job_id}")
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


