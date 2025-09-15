from fastapi import Request, HTTPException
from loguru import logger
from webhook import webhook_sender, WebhookPayload
import asyncio
import os


# Dependency to get services
def get_services(request:Request):
    if not all([request.app.state.asr_service, request.app.state.audio_processor, request.app.state.batch_processor]):
        raise HTTPException(status_code=503, detail="Services not initialized")
    return request.app.state.asr_service, request.app.state.audio_processor, request.app.state.batch_processor

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
