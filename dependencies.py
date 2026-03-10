from fastapi import Request, Depends
from loguru import logger
from webhook import webhook_sender, WebhookPayload
import asyncio
from generated.prisma import Prisma
from db_services.transcription import TranscriptionJobService
from db_services.auth import PrismaAPIKeyStore
from db_services.user import UserService
from intelligent_pipeline.pipeline import Pipeline
from typing import Optional

async def run_async_pipeline(audio_path: str, request_id: str, callback = None, call_metadata: dict = None):
    """Convenience function to run the pipeline synchronously (e.g. for scripts)"""
    pipeline = Pipeline()
    raw_transcription, labeled_transcription, analysis, metadata = await pipeline.run_pipeline(audio_path, request_id, call_metadata)
    if callback:
        callback(raw_transcription, labeled_transcription, analysis, metadata)

async def process_audio_background_db(
    job_id: str,
    audio_path: str,
    enhance_audio: bool,
    remove_silence: bool,
    priority: int,
    callback_url: Optional[str] = None,
    include_intelligence: bool = False,
    intelligence_mode: str = "auto_detect",
    api_key: str = None,
    transcription_svc = None,
    call_metadata: dict = None
):
    """Background task for async audio processing with database integration"""
    from datetime import datetime, timezone
    
    try:
        # Update job status to processing and get job for createdAt
        job = None
        if transcription_svc:
            job = await transcription_svc.update_job_status(job_id, 'processing')

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        def pipeline_callback(raw_trans, labeled_trans, analysis, diar_info):
            async def _handle_results():
                # Calculate processing time from job creation
                end_time = datetime.now(timezone.utc)
                processing_time = 0.0
                if job and job.createdAt:
                    # Ensure createdAt is timezone-aware
                    created_at = job.createdAt if job.createdAt.tzinfo else job.createdAt.replace(tzinfo=timezone.utc)
                    processing_time = (end_time - created_at).total_seconds()
                
                # Save result to DB
                intelligence_result = {}
                if transcription_svc:
                    if analysis:
                        try:
                            import json
                            intelligence_result = json.loads(analysis) if isinstance(analysis, str) else analysis
                        except:
                            intelligence_result = {"raw": analysis}

                    await transcription_svc.save_transcription_result(
                        job_id=job_id,
                        text=raw_trans,
                        diarization={'conversation':labeled_trans, "info":diar_info},
                        intelligence=intelligence_result,
                        confidence=1.0, 
                        rtf=0.0,
                        processing_time=processing_time,
                        chunks=diar_info.get('chunk_count', 0)
                    )

                if api_key:
                    try:
                        from db_services.auth import update_audio_usage
                        from utils import get_audio_duration

                        await update_audio_usage(api_key, get_audio_duration(audio_path))
                    except Exception as usage_exc:
                        logger.warning(f"Failed to update audio usage for job {job_id}: {usage_exc}")

                # Webhook (optional for streaming sessions where DB is primary sink)
                if callback_url:
                    webhook_payload = WebhookPayload(
                        job_id=job_id,
                        transcription=raw_trans,
                        ai_analysis=intelligence_result.get("analysis"),
                        diarized_transcription=labeled_trans,
                        agent_tasks=intelligence_result.get("agent_tasks")
                    )
                    await webhook_sender.send_webhook(callback_url, webhook_payload)
                else:
                    logger.info(f"No callback_url configured for job {job_id}; skipping webhook delivery")

            # Schedule it properly from thread
            asyncio.run_coroutine_threadsafe(_handle_results(), loop)


        # Run pipeline in a separate thread to avoid blocking main loop
        # run_async_pipeline is synchronous and uses asyncio.run() internally

        await run_async_pipeline(audio_path, job_id, pipeline_callback, call_metadata)
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        
        # Update job status to failed
        if transcription_svc:
            await transcription_svc.update_job_status(job_id, 'failed', error_message=str(e))
        
        # Send error webhook when callback target is configured
        if callback_url:
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
def get_transcription_service(
    db = Depends(get_db)
) -> TranscriptionJobService:
    return TranscriptionJobService(db)

# Dependency to get auth key service
def get_auth_service(
    db = Depends(get_db)
) -> PrismaAPIKeyStore:
    return PrismaAPIKeyStore(db)

def get_user_service(
    db = Depends(get_db)
) -> UserService:
    return UserService(db)
