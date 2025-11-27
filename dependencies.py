from fastapi import Request, HTTPException
from loguru import logger
from webhook import webhook_sender, WebhookPayload
import asyncio
import os
from generated.prisma import Prisma
from services.diarization_service import DiarizationService, store_diar_segments
from config import get_diarization_settings
import json

# Dependency to get services
def get_services(request:Request):
    if not all([request.app.state.asr_service, request.app.state.batch_processor]):
        raise HTTPException(status_code=503, detail="Services not initialized")
    return request.app.state.asr_service, request.app.state.batch_processor

def triton_asr_callback(transcription_svc, triton_svc, webhook_sender=None, include_intelligence: bool = False):
    """
    Thread-safe callback factory for Triton async_infer.

    - transcription_svc: DB service instance
    - triton_svc: service exposing process_intelligence(...)
    - webhook_sender: optional object (falls back to module-level import). May be sync or async.
    - include_intelligence: whether to call intelligence model
    """
    ws = webhook_sender if webhook_sender is not None else global_webhook_sender

    # Capture the main event loop at creation time. This must be created while the loop is running.
    try:
        main_loop = asyncio.get_running_loop()
    except RuntimeError:
        # If not created from within a running loop, try get_event_loop (less ideal)
        main_loop = asyncio.get_event_loop()

    async def handle_transcription_result(result, request_id):
        """Coroutine that runs on the main loop to handle a successful inference result."""
        try:
            # Robust transcription extraction
            transcription = None
            try:
                arr = result.as_numpy("transcription")
                # try several index shapes
                if arr is not None:
                    if arr.size == 0:
                        transcription = None
                    else:
                        v = arr.flatten()[0]
                        if isinstance(v, (bytes, bytearray)):
                            transcription = v.decode("utf-8")
                        else:
                            transcription = str(v)
            except Exception:
                # fallback fields
                transcription = getattr(result, "transcription", None) or getattr(result, "text", None)

            if not transcription:
                raise RuntimeError("Failed to extract transcription from Triton result")

            intelligence_result = {}

            # Intelligence step (runs via Triton service which itself may be async/callback-based)
            if include_intelligence and triton_svc:
                # Create a future bound to the main loop
                loop = asyncio.get_running_loop()
                intel_future = loop.create_future()

                def intelligence_callback(result, error):
                    # Called from Triton's thread — schedule set_result on main loop
                    print("INTELLIGENCE CALLBACK", result)
                    def _set_result():
                        if intel_future.done():
                            return
                        if error:
                            logger.error(f"[INTELLIGENCE] Error for job {request_id}: {error}")
                            intel_future.set_result({})
                            return
                        try:
                            arr = result.as_numpy("generated_text")
                            if arr is not None and arr.size:
                                text_blob = arr.flatten()[0]
                                if isinstance(text_blob, (bytes, bytearray)):
                                    text_blob = text_blob.decode("utf-8")
                                try:
                                    intel_future.set_result(json.loads(text_blob))
                                except Exception:
                                    intel_future.set_result({"generated_text": text_blob})
                            else:
                                # fallback
                                text_blob = getattr(result, "generated_text", None)
                                intel_future.set_result({"generated_text": text_blob} if text_blob else {})
                        except Exception as e:
                            logger.exception(f"[INTELLIGENCE] parse failed for {request_id}: {e}")
                            if not intel_future.done():
                                intel_future.set_result({})

                    # schedule on main loop
                    try:
                        loop.call_soon_threadsafe(_set_result)
                    except RuntimeError:
                        # fallback: use run_coroutine_threadsafe to set result
                        try:
                            asyncio.run_coroutine_threadsafe(_set_result(), loop)
                        except Exception:
                            pass

                # Start intelligence job (non-blocking). Expect triton_svc.process_intelligence to accept on_complete callback.
                try:
                    triton_svc.process_intelligence(
                        prompt=transcription,
                        request_id=request_id,
                        on_complete=intelligence_callback
                    )
                    intelligence_result = await intel_future
                except Exception as e:
                    logger.error(f"Failed to run intelligence for {request_id}: {e}")
                    intelligence_result = {}

            # Save transcription + intelligence
            try:
                await transcription_svc.save_transcription_result(
                    job_id=request_id,
                    text=transcription,
                    confidence=1.0,
                    rtf=1.0,
                    processing_time=0,
                    chunks=1,
                    diarization={},
                    intelligence=intelligence_result
                )
            except Exception as e:
                logger.error(f"Failed to save transcription for {request_id}: {e}")

            # Send webhook (handle sync vs async send_webhook)
            try:
                callback_url = await transcription_svc.get_callback_url(request_id)
            except Exception:
                callback_url = None

            if callback_url:
                payload = WebhookPayload(
                    job_id=request_id,
                    status="completed",
                    result={"transcription": transcription},
                    intelligence_data=intelligence_result
                )
                send_fn = getattr(ws, "send_webhook", None)
                if send_fn is None:
                    logger.error("webhook sender has no send_webhook method")
                else:
                    try:
                        if asyncio.iscoroutinefunction(send_fn):
                            await send_fn(callback_url, payload)
                        else:
                            # run sync webhook send in threadpool
                            await asyncio.to_thread(send_fn, callback_url, payload)
                    except Exception as e:
                        logger.error(f"Failed to send success webhook for job {request_id}: {e}")

            # update job status to completed
            try:
                await transcription_svc.update_job_status(request_id, "completed")
            except Exception as e:
                logger.error(f"Failed to update job status completed for {request_id}: {e}")

            print(json.dumps({
                "request_id": request_id,
                "success": True,
                "transcription": transcription,
                "intelligence": intelligence_result
            }, indent=2))

        except Exception as e:
            logger.exception(f"Error handling transcription result for {request_id}: {e}")
            # Update job status to failed
            try:
                await transcription_svc.update_job_status(request_id, "failed", error_message=str(e))
            except Exception:
                pass

            # Send failure webhook safely
            try:
                callback_url = await transcription_svc.get_callback_url(request_id)
                if callback_url:
                    payload = WebhookPayload(job_id=request_id, status="failed", error=str(e))
                    send_fn = getattr(ws, "send_webhook", None)
                    if send_fn:
                        if asyncio.iscoroutinefunction(send_fn):
                            await send_fn(callback_url, payload)
                        else:
                            await asyncio.to_thread(send_fn, callback_url, payload)
            except Exception as webhook_err:
                logger.error(f"Failed to send failure webhook for job {request_id}: {webhook_err}")

    # This is the *synchronous* callback that Triton will call from a gRPC thread.
    def callback(result, error):
        # Robust request_id extraction
        try:
            request_id = getattr(result, "id", None)
            if request_id is None:
                try:
                    resp = result.get_response()
                    request_id = getattr(resp, "id", None)
                except Exception:
                    request_id = "unknown"
        except Exception:
            request_id = "unknown"

        if error:
            # Schedule error handling coroutine on the main loop
            async def _handle_error():
                try:
                    await transcription_svc.update_job_status(request_id, "failed", error_message=str(error))
                except Exception:
                    pass
                try:
                    callback_url = await transcription_svc.get_callback_url(request_id)
                    if callback_url:
                        payload = WebhookPayload(job_id=request_id, status="failed", error=str(error))
                        send_fn = getattr(ws, "send_webhook", None)
                        if send_fn:
                            if asyncio.iscoroutinefunction(send_fn):
                                await send_fn(callback_url, payload)
                            else:
                                await asyncio.to_thread(send_fn, callback_url, payload)
                except Exception as webhook_err:
                    logger.error(f"Failed to send failure webhook for job {request_id}: {webhook_err}")

            # use run_coroutine_threadsafe to schedule from Triton thread
            try:
                asyncio.run_coroutine_threadsafe(_handle_error(), main_loop)
            except Exception as e:
                logger.exception(f"Failed to schedule error handler for {request_id}: {e}")

        else:
            # Schedule successful result handler on main loop
            try:
                asyncio.run_coroutine_threadsafe(handle_transcription_result(result, request_id), main_loop)
            except Exception as e:
                logger.exception(f"Failed to schedule success handler for {request_id}: {e}")

    return callback

def normalize_for_triton(audio_path: str) -> str:
    # If you store files as "uploads/2025-11-22/...", Triton model expects "2025-11-22/..."
    if audio_path.startswith("uploads/"):
        return audio_path[len("uploads/"):].lstrip("/")
    # If absolute or already normalized, return as-is
    return audio_path

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
    transcription_svc = None,
    triton_svc = None
):
    """Background task for async audio processing with database integration"""
    from datetime import datetime, timezone
    
    try:
        # Update job status to processing
        if transcription_svc:
            await transcription_svc.update_job_status(job_id, 'processing', started_at=datetime.now(timezone.utc))
        
        if triton_svc:
            callback = triton_asr_callback(
            transcription_svc=transcription_svc,
            triton_svc=triton_svc,
            webhook_sender=webhook_sender,
            include_intelligence=include_intelligence
        )
        triton_path = normalize_for_triton(audio_path)
        triton_svc.process_asr(triton_path, job_id, callback)
        
        
        # # Submit to Redis-based batch processor
        # result = await batch_proc.submit_job(
        #     job_id=job_id,
        #     audio_path=audio_path,
        #     include_intelligence=include_intelligence,
        #     callback_url=callback_url,
        # )

        # # Job is processing asynchronously - webhook will be sent when complete
        # if result and result.get('status') == 'queued':
        #     logger.info(f"Job {job_id} submitted to Redis queue: {result.get('num_chunks')} chunks")
        #     # Launch diarization in the background (mandatory)
        #     async def _run_diar():
        #         try:
        #             diar_cfg = get_diarization_settings()
        #             diar = DiarizationService(model_name=diar_cfg.model_name, max_speakers=diar_cfg.max_speakers)
                    
        #             logger.info(f"Starting diarization for job {job_id}")
        #             segments = await diar.run(audio_path, max_speakers=diar_cfg.max_speakers)
        #             num_spk = len(sorted({s.speaker for s in segments}))
                    
        #             await store_diar_segments(batch_proc.redis_client, job_id, segments, num_spk)
        #             logger.info(f"Diarization completed for job {job_id}: {len(segments)} segments, {num_spk} speakers")
                    
        #         except Exception as e:
        #             logger.error(f"Diarization failed for job {job_id}: {e}")
        #             # Store failure status in Redis so chunk_worker knows it failed
        #             try:
        #                 await batch_proc.redis_client.set(
        #                     f"diar:{job_id}:status",
        #                     "failed",
        #                     ex=3600  # Expire after 1 hour
        #                 )
        #             except Exception:
        #                 pass
            
        #     asyncio.create_task(_run_diar())
        #     return  # Exit early; completion orchestrated after stitching+diar

        # # For failed submissions, send error webhook
        # if not result or result.get('status') == 'failed':
        #     error_msg = result.get('error', 'Failed to submit job to queue') if result else 'Submission failed'
        #     if transcription_svc:
        #         await transcription_svc.update_job_status(job_id, 'failed', error_message=error_msg)

        #     webhook_payload = WebhookPayload(
        #         job_id=job_id,
        #         status="failed",
        #         error=error_msg
        #     )
        #     asyncio.create_task(webhook_sender.send_webhook(callback_url, webhook_payload))
        
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

# Dependency function to retrieve the initialized TritonService instance.
def get_triton_service(request: Request):
    triton = getattr(request.app.state, "triton_service", None)

    if triton is None:
        raise HTTPException(
            status_code=500,
            detail="Triton Inference Service is not initialized."
        )

    return triton