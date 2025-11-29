from fastapi import Request, HTTPException
from loguru import logger
from webhook import webhook_sender, WebhookPayload
import asyncio
import os
from generated.prisma import Prisma
from services.triton.triton_service import TritonService, run_async_pipeline


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
        
        # Define callback for result handling
        def pipeline_callback(raw_trans, labeled_trans, analysis, diar_info):
            async def _handle_results():
                    # Save result to DB
                    if transcription_svc:
                        # Extract intelligence result if available
                        intelligence_result = None
                        if analysis and isinstance(analysis, str):
                            try:
                                import json
                                intelligence_result = json.loads(analysis)
                            except:
                                intelligence_result = {"raw": analysis}
                        elif analysis:
                            intelligence_result = analysis

                  
                        
                        diarization_data = {
                            'numSpeakers': diar_info.get('total_speakers', 0),
                            'diarizationStatus': 'completed',
                            'audioDuration': diar_info.get('audio_duration', 0),
                            'info': diar_info 
                        }

                        await transcription_svc.save_transcription_result(
                            job_id=job_id,
                            text=raw_trans,
                            diarization={'conversation':labeled_trans, "info":diar_info},
                            intelligence=intelligence_result,
                            confidence=1.0, 
                            rtf=0.0,
                            processing_time=0.0,
                            chunks=diar_info.get('chunk_count', 0)
                        )

                    # Send webhook

                    webhook_payload = WebhookPayload(
                        job_id=job_id,
                        transcription=raw_trans,
                        ai_analysis=intelligence_result['analysis'],
                        diarized_transcription=labeled_trans
                    )
                    await webhook_sender.send_webhook(callback_url, webhook_payload)

            # Run async logic synchronously within the thread
            asyncio.run(_handle_results())

        # Run pipeline in a separate thread to avoid blocking main loop
        # run_async_pipeline is synchronous and uses asyncio.run() internally
        loop = asyncio.get_event_loop()
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
def get_transcription_service(request: Request):
    from db_services.transcription import TranscriptionJobService
    db = request.app.state.db
    return TranscriptionJobService(db)