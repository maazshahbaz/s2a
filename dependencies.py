from fastapi import Request, HTTPException, Depends
from loguru import logger
from webhook import webhook_sender, WebhookPayload
import asyncio
import os
from generated.prisma import Prisma
from db_services.transcription import TranscriptionJobService
from db_services.auth import PrismaAPIKeyStore
from db_services.user import UserService
from services.triton.triton_service import run_async_pipeline


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
                if transcription_svc:
                    intelligence_result = None
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

                # Webhook
                webhook_payload = WebhookPayload(
                    job_id=job_id,
                    transcription=raw_trans,
                    ai_analysis=intelligence_result.get("analysis"),
                    diarized_transcription=labeled_trans
                )
                await webhook_sender.send_webhook(callback_url, webhook_payload)

            # Schedule it properly from thread
            asyncio.run_coroutine_threadsafe(_handle_results(), loop)


        # Run pipeline in a separate thread to avoid blocking main loop
        # run_async_pipeline is synchronous and uses asyncio.run() internally

        await loop.run_in_executor(None, run_async_pipeline, audio_path, job_id, pipeline_callback)
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