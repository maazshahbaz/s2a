from fastapi import Request, HTTPException
from loguru import logger
from webhook import webhook_sender, WebhookPayload
import asyncio
import os
from generated.prisma import Prisma
from services.diarization_service import DiarizationService, store_diar_segments
from config import get_diarization_settings


# Dependency to get services
def get_services(request:Request):
    if not all([request.app.state.asr_service, request.app.state.batch_processor]):
        raise HTTPException(status_code=503, detail="Services not initialized")
    return request.app.state.asr_service, request.app.state.batch_processor

async def process_audio_background_db(
    job_id: str,
    audio_path: str,
    enhance_audio: bool,
    remove_silence: bool,
    priority: int,
    callback_url: str,
    asr_svc,
    batch_proc,
    include_intelligence: bool = False,
    intelligence_mode: str = "auto_detect",
    api_key: str = None,
    transcription_svc = None
):
    """Background task for async audio processing with database integration"""
    from datetime import datetime, timezone
    
    try:
        # Update job status to processing
        if transcription_svc:
            await transcription_svc.update_job_status(job_id, 'processing', started_at=datetime.now(timezone.utc))
        
        # Submit to Redis-based batch processor
        result = await batch_proc.submit_job(
            job_id=job_id,
            audio_path=audio_path,
            callback_url=callback_url
        )

        # Job is processing asynchronously - webhook will be sent when complete
        if result and result.get('status') == 'queued':
            logger.info(f"Job {job_id} submitted to Redis queue: {result.get('num_chunks')} chunks")
            # Launch diarization in the background (mandatory)
            async def _run_diar():
                try:
                    diar_cfg = get_diarization_settings()
                    diar = DiarizationService(model_name=diar_cfg.model_name, max_speakers=diar_cfg.max_speakers)
                    
                    logger.info(f"Starting diarization for job {job_id}")
                    segments = await diar.run(audio_path, max_speakers=diar_cfg.max_speakers)
                    num_spk = len(sorted({s.speaker for s in segments}))
                    
                    await store_diar_segments(batch_proc.redis_client, job_id, segments, num_spk)
                    logger.info(f"Diarization completed for job {job_id}: {len(segments)} segments, {num_spk} speakers")
                    
                except Exception as e:
                    logger.error(f"Diarization failed for job {job_id}: {e}")
                    # Store failure status in Redis so chunk_worker knows it failed
                    try:
                        await batch_proc.redis_client.set(
                            f"diar:{job_id}:status",
                            "failed",
                            ex=3600  # Expire after 1 hour
                        )
                    except Exception:
                        pass
            
            asyncio.create_task(_run_diar())
            return  # Exit early; completion orchestrated after stitching+diar

        # For failed submissions, send error webhook
        if not result or result.get('status') == 'failed':
            error_msg = result.get('error', 'Failed to submit job to queue') if result else 'Submission failed'
            if transcription_svc:
                await transcription_svc.update_job_status(job_id, 'failed', error_message=error_msg)

            webhook_payload = WebhookPayload(
                job_id=job_id,
                status="failed",
                error=error_msg
            )
            asyncio.create_task(webhook_sender.send_webhook(callback_url, webhook_payload))
        
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