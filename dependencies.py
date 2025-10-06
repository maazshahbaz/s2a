from fastapi import Request, HTTPException
from loguru import logger
from webhook import webhook_sender, WebhookPayload
import asyncio
import os
from generated.prisma import Prisma


# Dependency to get services
def get_services(request:Request):
    if not all([request.app.state.asr_service, request.app.state.audio_processor, request.app.state.batch_processor]):
        raise HTTPException(status_code=503, detail="Services not initialized")
    return request.app.state.asr_service, request.app.state.audio_processor, request.app.state.batch_processor

async def process_audio_background_db(
    job_id: str,
    audio_path: str,
    enhance_audio: bool,
    remove_silence: bool,
    priority: int,
    callback_url: str,
    asr_svc,
    audio_proc,
    batch_proc,
    api_key: str = None,
    transcription_svc = None
):
    """Background task for async audio processing with database integration"""
    from datetime import datetime, timezone
    
    try:
        # Update job status to processing
        if transcription_svc:
            await transcription_svc.update_job_status(job_id, 'processing', started_at=datetime.now(timezone.utc))
        
        # Process audio
        audio, sr, audio_info = audio_proc.process_audio_file(
            audio_path,
            enhance=enhance_audio,
            validate=True
        )
        
        # Check minimum duration (both sync/async same rule)
        if audio_info['duration'] < asr_svc.min_audio_duration:
            error_msg = f"Audio too short ({audio_info['duration']:.1f}s < {asr_svc.min_audio_duration:.1f}s)"
            logger.info(f"Job {job_id}: {error_msg}")
            if transcription_svc:
                await transcription_svc.update_job_status(job_id, 'rejected', error_message=error_msg)
            return
        
        # Check maximum duration for ASYNC API (2 hours max)
        from config import get_settings
        settings = get_settings()
        if audio_info['duration'] > settings.max_async_audio_duration:
            error_msg = f"Audio too long ({audio_info['duration']:.1f}s > {settings.max_async_audio_duration:.1f}s)"
            logger.info(f"Job {job_id}: {error_msg}")
            if transcription_svc:
                await transcription_svc.update_job_status(job_id, 'rejected', error_message=error_msg)
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
            # Save transcription result to database (this also updates job status to completed)
            if transcription_svc:
                await transcription_svc.save_transcription_result(
                    job_id=job_id,
                    text=result.get('text', ''),
                    confidence=result.get('confidence'),
                    rtf=result.get('rtf'),
                    processing_time=result.get('processing_time'),
                    chunks=1,
                    audio_quality=audio_info.get('quality_metrics')
                )
            
            # Track audio duration usage for billing
            if api_key:
                from db_services.auth import update_audio_usage
                await update_audio_usage(api_key, audio_info['duration'])
            
            webhook_payload = WebhookPayload(
                job_id=job_id,
                status="completed",
                result=result,
                processing_time=result.get('processing_time')
            )
        else:
            error_msg = "Transcription processing failed"
            if transcription_svc:
                await transcription_svc.update_job_status(job_id, 'failed', error_message=error_msg)
            
            webhook_payload = WebhookPayload(
                job_id=job_id,
                status="failed",
                error=error_msg
            )
        
        # Send webhook asynchronously (don't wait for it)
        asyncio.create_task(webhook_sender.send_webhook(callback_url, webhook_payload))
        
        logger.info(f"Job {job_id} completed and webhook sent to {callback_url}")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        
        # Update job status to failed
        if transcription_svc:
            await transcription_svc.update_job_status(job_id, 'failed', error_message=str(e))
        
        # Send error webhook
        error_payload = WebhookPayload(
            job_id=job_id,
            status="failed",
            error=str(e)
        )
        asyncio.create_task(webhook_sender.send_webhook(callback_url, error_payload))
    
    finally:
        # Clean up audio file (keep it for completed jobs, remove for failed/rejected)
        # For production, you might want to implement a cleanup job that removes old files
        pass

# Dependency to get DB
async def get_db(request: Request) -> Prisma:
    return request.app.state.db

# Dependency to get transcription service
def get_transcription_service(request: Request):
    from db_services.transcription import TranscriptionJobService
    db = request.app.state.db
    return TranscriptionJobService(db)