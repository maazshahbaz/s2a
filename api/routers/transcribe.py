from fastapi import APIRouter, Response, UploadFile, File, Depends, Request, HTTPException, Form, BackgroundTasks
from api.schemas import TranscriptionResponse, TranscribeAsyncResponse, StatusResponse
import os
from dependencies import get_services
from api.schemas import  TranscriptionResponse, TranscribeAsyncResponse, StatusResponse
from dependencies import get_services, get_transcription_service
from db_services.auth import require_permission, update_request_usage, get_rate_limit_headers, APIKey
from db_services.transcription import store_uploaded_file, delete_audio_file
import uuid
from config import get_settings
from loguru import logger
from webhook import webhook_sender
from dependencies import process_audio_background_db

router = APIRouter(prefix="/transcribe", tags=["Transcription"])

@router.post("", response_model=TranscribeAsyncResponse)
async def transcribe_async(
    request: Request,
    response: Response,
    callback_url: str = Form(...),
    audio_file: UploadFile = File(...),
    enhance_audio: bool = True,
    remove_silence: bool = False,
    include_intelligence: bool = True,
    intelligence_mode: str = "auto_detect",
    priority: int = 0,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    key_info: APIKey = Depends(require_permission("transcribe")),
    services = Depends(get_services),
    transcription_svc = Depends(get_transcription_service)
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
    
    # Store uploaded file permanently
    audio_path = await store_uploaded_file(audio_file, job_id)
    
    try:
        # Get audio info for duration
        audio, sr, audio_info = audio_proc.process_audio_file(
            audio_path,
            enhance=enhance_audio,
            validate=True
        )

         # Check minimum duration (both sync/async same rule)
        if audio_info['duration'] < asr_svc.min_audio_duration:
            raise HTTPException(status_code=400, detail="Audio too short. Minimum duration is 1 second.")
        
        
        # Check maximum duration for SYNC API (2 minutes max)
        settings = get_settings()
        if audio_info['duration'] > settings.max_audio_duration:
            error_msg = f"Audio too long for sync API. Maximum duration is {settings.max_audio_duration/60:.0f} minutes."
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Create job record in database
        job = await transcription_svc.create_job(
            job_id=job_id,
            audio_path=audio_path,
            is_async=True,
            enhance_audio=enhance_audio,
            remove_silence=remove_silence,
            priority=priority,
            callback_url=callback_url,
            audio_duration=audio_info['duration']
        )
        
        # Add to background processing
        background_tasks.add_task(
            process_audio_background_db,
            job_id, audio_path, enhance_audio, remove_silence, priority, callback_url,
            asr_svc, audio_proc, batch_proc, include_intelligence, intelligence_mode,
            request.state.api_key, transcription_svc
        )
        
        return TranscribeAsyncResponse(job_id=job_id, status="accepted")
    except HTTPException:
        try:
            if os.path.exists(audio_path):
                await delete_audio_file(audio_path)
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up audio file {audio_path}: {cleanup_err}")
        raise    
    except Exception as e:
        logger.error(f"Error setting up async job {job_id}: {e}")
        # Update job status to failed if it was created
        try:
            await transcription_svc.update_job_status(job_id, 'failed', error_message=str(e))
        except:
            pass
        # Delete file only on failure
        try:
            if os.path.exists(audio_path):
                await delete_audio_file(audio_path)
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up audio file {audio_path}: {cleanup_err}")
        raise HTTPException(status_code=500, detail=f"Failed to set up async transcription: {str(e)}")

@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_transcription_status(
    job_id: str,
    request: Request,
    response: Response,
    key_info: APIKey = Depends(require_permission("status")),
    services = Depends(get_services),
    transcription_svc = Depends(get_transcription_service)
):
    """Get status of async transcription job - requires status permission"""
    asr_svc, audio_proc, batch_proc = services
    
    # Add rate limit headers
    headers = get_rate_limit_headers(request)
    for key, value in headers.items():
        response.headers[key] = value
    
    # Get job from database
    job = await transcription_svc.get_job(job_id)
    
    if job is None:
        return StatusResponse(job_id=job_id, status="not_found")
    
    # Return status based on job status
    if job.status == "failed":
        return StatusResponse(
            job_id=job_id,
            status="failed",
            error=job.errorMessage
        )
    elif job.status == "rejected":
        return StatusResponse(
            job_id=job_id,
            status="rejected",
            error=job.errorMessage
        )
    elif job.status == "completed" and job.transcriptionResult:
        # Job is completed with results
        result = job.transcriptionResult
        transcription_response = TranscriptionResponse(
            job_id=job_id,
            status="completed",
            text=result.text,
            duration=job.audioDuration or 0,
            rtf=result.rtf or 0,
            processing_time=result.processingTime or 0,
            chunks=result.chunks or 1,
            confidence=result.confidence,
            audio_quality=result.audioQuality
        )
        
        return StatusResponse(
            job_id=job_id,
            status="completed",
            result=transcription_response
        )
    else:
        # Job is pending or processing
        return StatusResponse(job_id=job_id, status=job.status)


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


