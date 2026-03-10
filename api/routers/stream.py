import asyncio
import json
import os
import re
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from loguru import logger

from api.streaming_security import is_ip_allowed, parse_allowed_networks
from config import get_settings
from webhook import webhook_sender

router = APIRouter(tags=["streaming"])
SENTENCE_SPLIT_RE = re.compile(r".+?(?:[.!?]+)(?=\s|$)")
FORCED_SENTENCE_WORDS = 14


def _canonical_token(token: str) -> str:
    """Normalize a token for overlap matching (case/punctuation-insensitive)."""
    core = re.sub(r"^\W+|\W+$", "", token or "").lower()
    return core or (token or "").lower()


def _find_suffix_prefix_word_overlap(committed_words, new_words, max_window: int = 80) -> int:
    """
    Find the largest K where committed_words[-K:] == new_words[:K] under canonical token matching.
    """
    if not committed_words or not new_words:
        return 0

    committed_norm = [_canonical_token(w) for w in committed_words]
    new_norm = [_canonical_token(w) for w in new_words]
    max_overlap = min(len(committed_norm), len(new_norm), max_window)

    for k in range(max_overlap, 0, -1):
        if committed_norm[-k:] == new_norm[:k]:
            return k
    return 0


def _extract_incremental_delta(model_text: str, committed_text: str):
    """
    Merge model text into a running committed transcript and return only the new delta.

    Handles sparse/repeated/non-cumulative streaming outputs where the model may emit:
      - empty text on some chunks
      - repeated short hypotheses
      - overlapping fragments
    """
    new_text = re.sub(r"\s+", " ", (model_text or "").strip())
    committed = re.sub(r"\s+", " ", (committed_text or "").strip())

    if not new_text:
        return "", committed
    if not committed:
        return new_text, new_text

    lower_new = new_text.lower()
    lower_committed = committed.lower()

    # Fully duplicate or already-covered text.
    if lower_new == lower_committed:
        return "", committed
    if lower_committed.endswith(lower_new):
        return "", committed
    if f" {lower_new} " in f" {lower_committed} ":
        return "", committed

    # Model returned true cumulative text extension.
    if lower_new.startswith(lower_committed):
        delta = new_text[len(committed):].strip()
        if not delta or not any(ch.isalnum() for ch in delta):
            return "", committed
        return delta, _append_live_text(committed, delta)

    # Model returned a partially overlapping fragment.
    committed_words = committed.split()
    new_words = new_text.split()
    overlap = _find_suffix_prefix_word_overlap(committed_words, new_words)
    if overlap >= len(new_words):
        return "", committed
    if overlap > 0:
        delta = " ".join(new_words[overlap:]).strip()
        if not delta or not any(ch.isalnum() for ch in delta):
            return "", committed
        return delta, _append_live_text(committed, delta)

    # No overlap: treat as fresh continuation.
    if not any(ch.isalnum() for ch in new_text):
        return "", committed
    return new_text, _append_live_text(committed, new_text)


def _append_live_text(existing_text: str, new_text: str) -> str:
    """Append new text to the live sentence buffer with normalized spacing."""
    clean_new = re.sub(r"\s+", " ", (new_text or "").strip())
    if not clean_new:
        return (existing_text or "").strip()
    if not existing_text:
        return clean_new
    return f"{existing_text.strip()} {clean_new}".strip()


def _extract_sentence_chunks(buffer_text: str, force_flush: bool = False):
    """
    Split buffered text into sentence-level chunks.
    Falls back to fixed-size chunks when punctuation is absent for too long.
    """
    normalized = re.sub(r"\s+", " ", (buffer_text or "").strip())
    if not normalized:
        return [], ""

    chunks = []
    last_end = 0
    for match in SENTENCE_SPLIT_RE.finditer(normalized):
        sentence = match.group().strip()
        if sentence:
            chunks.append(sentence)
        last_end = match.end()

    remainder = normalized[last_end:].strip()

    if force_flush and remainder:
        chunks.append(remainder)
        remainder = ""
    elif not chunks and remainder:
        words = remainder.split()
        if len(words) >= FORCED_SENTENCE_WORDS:
            chunks.append(" ".join(words[:FORCED_SENTENCE_WORDS]))
            remainder = " ".join(words[FORCED_SENTENCE_WORDS:])

    return chunks, remainder


def _choose_speaker_id(diar_result: dict, time_point: float) -> int:
    def _coerce_speaker_id(raw_speaker) -> int:
        if isinstance(raw_speaker, int):
            return raw_speaker
        if isinstance(raw_speaker, str):
            match = re.search(r"\d+", raw_speaker)
            if match:
                return int(match.group())
        try:
            return int(raw_speaker)
        except (TypeError, ValueError):
            return 0

    segments = (diar_result or {}).get("segments", [])
    parsed_segments = []
    parsed_speaker_ids = []

    for seg in segments:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            speaker = _coerce_speaker_id(seg.get("speaker", 0))
        except (TypeError, ValueError, AttributeError):
            continue
        parsed_segments.append((start, end, speaker))
        parsed_speaker_ids.append(speaker)

    if not parsed_segments:
        return 0

    # Some diarization outputs are 1-based labels (1,2,3...).
    # Normalize to 0-based IDs for consistent API payloads.
    speaker_base = 1 if min(parsed_speaker_ids) >= 1 else 0
    best_speaker = 0
    best_distance = float("inf")

    for start, end, raw_speaker in parsed_segments:
        speaker = max(0, raw_speaker - speaker_base)

        if start <= time_point <= end:
            return speaker
        distance = min(abs(time_point - start), abs(time_point - end))
        if distance < best_distance:
            best_distance = distance
            best_speaker = speaker

    return best_speaker


def _build_live_delta_segment(delta_text: str, diar_result: dict, cursor_time: float):
    """
    Build a single non-overlapping live segment with approximate timing.
    Returns (segment_or_none, new_cursor_time).
    """
    text = (delta_text or "").strip()
    if not text:
        return None, cursor_time

    words = max(1, len(text.split()))
    est_duration = max(0.25, words * 0.32)
    start = max(0.0, cursor_time)
    end = start + est_duration
    speaker_id = _choose_speaker_id(diar_result, (start + end) / 2.0)

    segment = {
        "speaker": f"Speaker {speaker_id + 1}",
        "speaker_id": speaker_id,
        "text": text,
        "start": round(start, 3),
        "end": round(end, 3),
    }
    return segment, end


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
            await websocket.send_json({"type": "error", "message": str(exc)})
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
        committed_transcript = ""
        emission_cursor = 0.0
        live_sentence_buffer = ""

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

                    delta_text, committed_transcript = _extract_incremental_delta(
                        model_text,
                        committed_transcript,
                    )
                    if not delta_text:
                        continue

                    live_sentence_buffer = _append_live_text(live_sentence_buffer, delta_text)
                    sentence_chunks, live_sentence_buffer = _extract_sentence_chunks(
                        live_sentence_buffer,
                        force_flush=False,
                    )
                    if not sentence_chunks:
                        continue

                    emission_cursor = max(emission_cursor, time_offset)
                    emitted_segments = []
                    for sentence_text in sentence_chunks:
                        segment, emission_cursor = _build_live_delta_segment(
                            sentence_text,
                            latest_diar_result,
                            emission_cursor,
                        )
                        if not segment:
                            continue
                        session.add_transcript_segment(segment)
                        emitted_segments.append(segment)

                    if emitted_segments:
                        await websocket.send_json(
                            {
                                "type": "transcript.update",
                                "is_final": False,
                                "cumulative_text": committed_transcript,
                                "segments": emitted_segments,
                                "update_mode": "sentence",
                                "num_speakers": max(
                                    latest_diar_result.get("num_speakers", 0),
                                    max(seg.get("speaker_id", 0) + 1 for seg in emitted_segments),
                                ),
                            }
                        )
                except asyncio.TimeoutError:
                    logger.error(f"[Stream] Inference timeout for {session_id}")
                    await websocket.send_json({"type": "error", "message": "Inference timeout"})
                    break
                except Exception as exc:
                    logger.error(f"[Stream] Inference error for {session_id}: {exc}")
                    await websocket.send_json({"type": "error", "message": f"Inference error: {str(exc)}"})

        remaining = session.get_remaining_chunk()
        if remaining is not None:
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
                    final_delta_text, committed_transcript = _extract_incremental_delta(
                        final_model_text,
                        committed_transcript,
                    )
                    if final_delta_text:
                        live_sentence_buffer = _append_live_text(live_sentence_buffer, final_delta_text)
            except asyncio.TimeoutError:
                logger.error(f"[Stream] Final chunk inference timeout for {session_id}")
            except Exception as exc:
                logger.error(f"[Stream] Final chunk error: {exc}")

        final_chunks, live_sentence_buffer = _extract_sentence_chunks(
            live_sentence_buffer,
            force_flush=True,
        )
        if final_chunks:
            final_segments = []
            for sentence_text in final_chunks:
                segment, emission_cursor = _build_live_delta_segment(
                    sentence_text,
                    latest_diar_result,
                    emission_cursor,
                )
                if not segment:
                    continue
                session.add_transcript_segment(segment)
                final_segments.append(segment)

            if final_segments:
                await websocket.send_json(
                    {
                        "type": "transcript.update",
                        "is_final": True,
                        "cumulative_text": committed_transcript,
                        "segments": final_segments,
                        "update_mode": "sentence",
                        "num_speakers": max(
                            latest_diar_result.get("num_speakers", 0),
                            max(seg.get("speaker_id", 0) + 1 for seg in final_segments),
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
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if session and session_manager:
            await session_manager.end_session(session.session_id)
