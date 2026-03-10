import asyncio
import json
import os
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from loguru import logger

from api.streaming_security import is_ip_allowed, parse_allowed_networks
from api.streaming_transcript import (
    append_transcript,
    build_chunk_segment,
    extract_model_delta,
)
from config import get_settings
from webhook import webhook_sender

router = APIRouter(tags=["streaming"])


async def _create_streaming_job_if_possible(
    app,
    settings,
    job_id: str,
    audio_path: str,
    callback_url: Optional[str],
    total_duration: float,
):
    """Create a DB-backed job record for streaming sessions when DB is available."""
    if settings.staging_mode:
        return None

    db = getattr(app.state, "db", None)
    if db is None:
        return None

    from db_services.transcription import TranscriptionJobService

    transcription_svc = TranscriptionJobService(db)
    audio_size = None
    try:
        if audio_path and os.path.exists(audio_path):
            audio_size = os.path.getsize(audio_path)
    except OSError:
        audio_size = None

    await transcription_svc.create_job(
        job_id=job_id,
        audio_path=audio_path,
        is_async=True,
        enhance_audio=True,
        remove_silence=False,
        priority=0,
        callback_url=callback_url,
        audio_duration=total_duration,
        audio_size=audio_size,
    )
    return transcription_svc


@router.get("/stream/health")
async def stream_health(request: Request):
    """
    Streaming readiness probe used by runtime monitors.
    """
    app = request.app
    session_manager = getattr(app.state, "session_manager", None)
    streaming_asr = getattr(app.state, "streaming_asr_client", None)
    streaming_diar = getattr(app.state, "streaming_diar_client", None)

    asr_ready = False
    diar_ready = False

    if streaming_asr and getattr(streaming_asr, "client", None):
        try:
            asr_ready = (
                await streaming_asr.client.is_server_live()
                and await streaming_asr.client.is_server_ready()
                and await streaming_asr.client.is_model_ready(streaming_asr.model_name)
            )
        except Exception:
            asr_ready = False

    if streaming_diar and getattr(streaming_diar, "client", None):
        try:
            diar_ready = (
                await streaming_diar.client.is_server_live()
                and await streaming_diar.client.is_server_ready()
                and await streaming_diar.client.is_model_ready(streaming_diar.model_name)
            )
        except Exception:
            diar_ready = False

    return {
        "status": "ok" if session_manager and asr_ready and diar_ready else "degraded",
        "session_manager_ready": session_manager is not None,
        "active_sessions": session_manager.active_count if session_manager else 0,
        "streaming_asr_ready": asr_ready,
        "streaming_diar_ready": diar_ready,
    }


@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming STT with diarization.

    Protocol:
        Client -> Server:
            1. JSON: {"type": "session.start", "session_id": "...", "call_metadata": {...},
                      "callback_url": "...", "audio_config": {"sample_rate": 8000}}
            2. Binary frames: raw PCM audio (16-bit signed LE)
            3. JSON: {"type": "session.end"}

        Server -> Client:
            1. JSON: {"type": "session.started", "session_id": "...", "job_id": "..."}
            2. JSON: {"type": "transcript.update", "segment_id": N,
                      "segments": [{"speaker": "Speaker 1", "text": "...", "start": 0.0, "end": 2.1}]}
            3. JSON: {"type": "session.ended", "total_duration": ..., "intelligence_status": "processing"}
    """
    await websocket.accept()

    settings = get_settings()
    session = None
    session_manager = None
    streaming_asr = None
    streaming_diar = None
    session_id = None
    callback_url = None
    call_metadata = {}
    api_key = None

    try:
        app = websocket.app
        session_manager = getattr(app.state, "session_manager", None)
        streaming_asr = getattr(app.state, "streaming_asr_client", None)
        streaming_diar = getattr(app.state, "streaming_diar_client", None)

        if not session_manager:
            await websocket.send_json({"type": "error", "message": "Streaming not available"})
            await websocket.close()
            return

        if not streaming_asr:
            await websocket.send_json({"type": "error", "message": "Streaming ASR backend unavailable"})
            await websocket.close()
            return

        if not streaming_diar:
            await websocket.send_json({"type": "error", "message": "Streaming diarization backend unavailable"})
            await websocket.close()
            return

        allowed_networks = parse_allowed_networks(settings.streaming_allowed_ips)
        client_ip = websocket.client.host if websocket.client else None
        if not is_ip_allowed(client_ip, allowed_networks):
            await websocket.send_json({"type": "error", "message": "Client IP is not allowed"})
            await websocket.close()
            return

        # Authentication in non-staging mode
        if not settings.staging_mode:
            token = websocket.query_params.get("token")
            if not token:
                await websocket.send_json(
                    {"type": "error", "message": "Authentication required. Pass ?token=bp-proj-xxx"}
                )
                await websocket.close()
                return

            try:
                from db_services.auth import api_key_store, rate_limiter

                if api_key_store is None:
                    await websocket.send_json({"type": "error", "message": "Authentication service unavailable"})
                    await websocket.close()
                    return

                key_info = await api_key_store.get_key(token)
                if not key_info or not key_info.is_active:
                    await websocket.send_json({"type": "error", "message": "Invalid or inactive API key"})
                    await websocket.close()
                    return
                if "transcribe" not in key_info.permissions:
                    await websocket.send_json({"type": "error", "message": "Permission 'transcribe' required"})
                    await websocket.close()
                    return

                rate_info = rate_limiter.check_rate_limit(key_info)
                if not rate_info.allowed:
                    await websocket.send_json({"type": "error", "message": "Rate limit exceeded"})
                    await websocket.close()
                    return

                await api_key_store.update_key_usage(token, audio_duration=0.0, track_request=True)
                api_key = token
            except Exception as exc:
                logger.error(f"[Stream] Auth error: {exc}")
                await websocket.send_json({"type": "error", "message": "Authentication failed"})
                await websocket.close()
                return

        # Wait for session.start
        try:
            start_msg = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=settings.streaming_session_start_timeout_seconds,
            )
        except asyncio.TimeoutError:
            await websocket.send_json({"type": "error", "message": "Timed out waiting for session.start"})
            await websocket.close()
            return

        if not isinstance(start_msg, dict):
            await websocket.send_json({"type": "error", "message": "session.start payload must be a JSON object"})
            await websocket.close()
            return

        if start_msg.get("type") != "session.start":
            await websocket.send_json({"type": "error", "message": "Expected session.start message"})
            await websocket.close()
            return

        session_id = start_msg.get("session_id", "")
        call_metadata = start_msg.get("call_metadata", {})
        if not isinstance(call_metadata, dict):
            call_metadata = {}

        callback_url = webhook_sender.sanitize_url(
            start_msg.get("callback_url") or settings.streaming_default_callback_url or ""
        )
        callback_url = callback_url or None
        if callback_url and not webhook_sender.validate_callback_url(callback_url):
            await websocket.send_json({"type": "error", "message": "Invalid callback_url"})
            await websocket.close()
            return

        audio_config = start_msg.get("audio_config", {})
        if audio_config is None:
            audio_config = {}
        if not isinstance(audio_config, dict):
            await websocket.send_json({"type": "error", "message": "audio_config must be a JSON object"})
            await websocket.close()
            return
        input_sample_rate = audio_config.get("sample_rate", 8000)
        try:
            input_sample_rate = int(input_sample_rate)
        except (TypeError, ValueError):
            await websocket.send_json({"type": "error", "message": "audio_config.sample_rate must be an integer"})
            await websocket.close()
            return

        if not session_id:
            await websocket.send_json({"type": "error", "message": "session_id is required"})
            await websocket.close()
            return
        if input_sample_rate <= 0 or input_sample_rate > 96000:
            await websocket.send_json({"type": "error", "message": "Unsupported sample_rate"})
            await websocket.close()
            return

        try:
            session = await session_manager.create_session(
                session_id=session_id,
                call_metadata=call_metadata,
                callback_url=callback_url,
                input_sample_rate=input_sample_rate,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning(f"[Stream] Session create failed for {session_id}: {exc}")
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Unable to create session (duplicate session_id or capacity reached)",
                }
            )
            await websocket.close()
            return

        await websocket.send_json(
            {
                "type": "session.started",
                "session_id": session_id,
                "job_id": session.job_id,
            }
        )
        logger.info(f"[Stream] Session {session_id} started (job_id={session.job_id})")

        latest_diar_result = {"segments": [], "num_speakers": 0}
        previous_model_text = ""
        cumulative_transcript = ""

        while True:
            if session.elapsed > settings.streaming_max_session_duration_seconds:
                await websocket.send_json({"type": "error", "message": "Session duration limit reached"})
                break

            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=settings.streaming_idle_timeout_seconds,
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "error", "message": "Session idle timeout"})
                break

            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    control_msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid control JSON"})
                    continue

                msg_type = control_msg.get("type")
                if msg_type == "session.end":
                    logger.info(f"[Stream] Session {session_id} ending (client requested)")
                    break
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                continue

            if "bytes" in message and message["bytes"] is not None:
                audio_bytes = message["bytes"]
                if not audio_bytes:
                    continue
                if len(audio_bytes) > settings.streaming_max_frame_bytes:
                    await websocket.send_json({"type": "error", "message": "Audio frame too large"})
                    break

                session.add_audio(audio_bytes)
                if session.current_ingress_bps() > settings.streaming_max_bytes_per_second:
                    await websocket.send_json({"type": "error", "message": "Audio ingress rate exceeded"})
                    break

                chunk = session.get_chunk()
                if chunk is None:
                    continue

                time_offset = session.get_time_offset()
                chunk_duration_seconds = len(chunk) / 16000.0
                chunk_end = time_offset + chunk_duration_seconds

                try:
                    asr_result, diar_result = await asyncio.wait_for(
                        asyncio.gather(
                            streaming_asr.transcribe_chunk(
                                chunk,
                                session_id,
                                16000,
                                request_id=f"{session_id}_asr_{session.chunks_processed}",
                            ),
                            streaming_diar.diarize_chunk(
                                chunk,
                                session_id,
                                16000,
                                request_id=f"{session_id}_diar_{session.chunks_processed}",
                            ),
                        ),
                        timeout=settings.streaming_inference_timeout_seconds,
                    )
                    # Diarization only updates every ~5s; keep latest result
                    if diar_result.get("diar_ran"):
                        latest_diar_result = diar_result

                    # streaming_asr may emit sparse/non-cumulative text; merge robustly.
                    model_text = asr_result.get("text", "").strip()
                    if not model_text:
                        continue

                    delta_text, previous_model_text = extract_model_delta(
                        model_text,
                        previous_model_text,
                    )
                    if not delta_text:
                        continue

                    updated_cumulative = append_transcript(cumulative_transcript, delta_text)
                    if updated_cumulative == cumulative_transcript:
                        continue
                    cumulative_transcript = updated_cumulative

                    segment = build_chunk_segment(
                        delta_text,
                        latest_diar_result,
                        time_offset,
                        chunk_end,
                    )
                    if not segment:
                        continue

                    session.add_transcript_segment(segment)
                    await websocket.send_json(
                        {
                            "type": "transcript.update",
                            "is_final": False,
                            "cumulative_text": cumulative_transcript,
                            "segments": [segment],
                            "update_mode": "sentence",
                            "num_speakers": max(
                                latest_diar_result.get("num_speakers", 0),
                                segment.get("speaker_id", 0) + 1,
                            ),
                        }
                    )
                except asyncio.TimeoutError:
                    logger.error(f"[Stream] Inference timeout for {session_id}")
                    await websocket.send_json({"type": "error", "message": "Inference timeout"})
                    break
                except Exception as exc:
                    logger.error(f"[Stream] Inference error for {session_id}: {exc}")
                    await websocket.send_json({"type": "error", "message": "Inference error"})

        remaining = session.get_remaining_chunk()
        final_segment = None
        if remaining is not None:
            remaining_offset = session.get_time_offset()
            remaining_end = remaining_offset + (len(remaining) / 16000.0)
            try:
                asr_result, diar_result = await asyncio.wait_for(
                    asyncio.gather(
                        streaming_asr.transcribe_chunk(
                            remaining,
                            session_id,
                            16000,
                            is_final=True,
                            request_id=f"{session_id}_final",
                        ),
                        streaming_diar.diarize_chunk(
                            remaining,
                            session_id,
                            16000,
                            is_final=True,
                            request_id=f"{session_id}_final_diar",
                        ),
                    ),
                    timeout=settings.streaming_inference_timeout_seconds,
                )
                if diar_result.get("diar_ran"):
                    latest_diar_result = diar_result

                final_model_text = asr_result.get("text", "").strip()
                if final_model_text:
                    final_delta_text, previous_model_text = extract_model_delta(
                        final_model_text,
                        previous_model_text,
                    )
                    if final_delta_text:
                        updated_cumulative = append_transcript(
                            cumulative_transcript,
                            final_delta_text,
                        )
                        if updated_cumulative == cumulative_transcript:
                            final_delta_text = ""
                        else:
                            cumulative_transcript = updated_cumulative
                    if final_delta_text:
                        final_segment = build_chunk_segment(
                            final_delta_text,
                            latest_diar_result,
                            remaining_offset,
                            remaining_end,
                        )
                        if final_segment:
                            session.add_transcript_segment(final_segment)
            except asyncio.TimeoutError:
                logger.error(f"[Stream] Final chunk inference timeout for {session_id}")
            except Exception as exc:
                logger.error(f"[Stream] Final chunk error: {exc}")

        if final_segment:
            await websocket.send_json(
                {
                    "type": "transcript.update",
                    "is_final": True,
                    "cumulative_text": cumulative_transcript,
                    "segments": [final_segment],
                    "update_mode": "sentence",
                    "num_speakers": max(
                        latest_diar_result.get("num_speakers", 0),
                        final_segment.get("speaker_id", 0) + 1,
                    ),
                }
            )

        audio_path = session.finalize()
        total_duration = session.duration
        intelligence_status = "skipped"

        if audio_path:
            try:
                transcription_svc = await _create_streaming_job_if_possible(
                    app=app,
                    settings=settings,
                    job_id=session.job_id,
                    audio_path=audio_path,
                    callback_url=callback_url,
                    total_duration=total_duration,
                )

                if callback_url or transcription_svc:
                    from dependencies import process_audio_background_db

                    asyncio.create_task(
                        process_audio_background_db(
                            job_id=session.job_id,
                            audio_path=audio_path,
                            enhance_audio=True,
                            remove_silence=False,
                            priority=0,
                            callback_url=callback_url,
                            include_intelligence=True,
                            intelligence_mode="full",
                            api_key=api_key,
                            transcription_svc=transcription_svc,
                            call_metadata=call_metadata,
                        )
                    )
                    intelligence_status = "processing"
                    logger.info(f"[Stream] Post-call pipeline started for {session_id}")
                else:
                    logger.warning(
                        f"[Stream] No callback_url and no DB service for session {session_id}; skipping post-call pipeline"
                    )
            except Exception as exc:
                logger.error(f"[Stream] Failed to start post-call pipeline: {exc}")
                intelligence_status = "failed"

        await websocket.send_json(
            {
                "type": "session.ended",
                "session_id": session_id,
                "job_id": session.job_id,
                "total_duration": round(total_duration, 2),
                "total_segments": session.segment_counter,
                "intelligence_status": intelligence_status,
            }
        )

        logger.info(
            f"[Stream] Session {session_id} completed: duration={total_duration:.1f}s, "
            f"segments={session.segment_counter}"
        )

    except WebSocketDisconnect:
        logger.info(f"[Stream] Client disconnected: {session.session_id if session else 'unknown'}")
    except Exception as exc:
        logger.error(f"[Stream] Unexpected error: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": "Unexpected server error"})
        except Exception:
            pass
    finally:
        if session and session_manager:
            await session_manager.end_session(session.session_id)
