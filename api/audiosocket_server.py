"""
AudioSocket TCP Server for Asterisk integration.

Asterisk AudioSocket protocol:
  - Frame format: 1 byte type + 2 bytes length (big-endian) + payload
  - Type 0x01: UUID (16 bytes, call identifier)
  - Type 0x10: Audio (PCM 16-bit LE)
  - Type 0x00: Hangup
  - Type 0x02: Silence
"""

import asyncio
import json
import os
import struct
import uuid
from typing import Optional

from loguru import logger

from api.streaming_security import is_ip_allowed, parse_allowed_networks
from api.streaming_transcript import (
    append_transcript,
    build_chunk_segment,
    extract_model_delta,
)
from webhook import webhook_sender

# AudioSocket frame types
FRAME_HANGUP = 0x00
FRAME_UUID = 0x01
FRAME_SILENCE = 0x02
FRAME_AUDIO = 0x10
FRAME_TRANSCRIPT_UPDATE = 0x20

HEADER_SIZE = 3  # 1 byte type + 2 bytes length


class AudioSocketServer:
    """TCP server implementing the Asterisk AudioSocket protocol."""

    def __init__(
        self,
        session_manager,
        streaming_asr,
        streaming_diar,
        port: int = 8003,
        default_callback_url: Optional[str] = None,
        staging_mode: bool = False,
        db=None,
        idle_timeout_seconds: float = 30.0,
        max_session_duration_seconds: float = 4 * 60 * 60,
        max_frame_bytes: int = 32768,
        max_bytes_per_second: int = 64000,
        inference_timeout_seconds: float = 20.0,
        allowed_ips: Optional[str] = None,
    ):
        self.session_manager = session_manager
        self.streaming_asr = streaming_asr
        self.streaming_diar = streaming_diar
        self.port = port
        self.staging_mode = staging_mode
        self.db = db
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_session_duration_seconds = max_session_duration_seconds
        self.max_frame_bytes = max_frame_bytes
        self.max_bytes_per_second = max_bytes_per_second
        self.inference_timeout_seconds = inference_timeout_seconds
        self.allowed_networks = parse_allowed_networks(allowed_ips)
        self._server = None

        sanitized_callback = webhook_sender.sanitize_url(default_callback_url or "")
        if sanitized_callback and webhook_sender.validate_callback_url(sanitized_callback):
            self.default_callback_url = sanitized_callback
        else:
            self.default_callback_url = None
            if sanitized_callback:
                logger.warning(
                    f"[AudioSocket] Ignoring invalid default callback URL: {sanitized_callback}"
                )

    async def start(self):
        """Start the AudioSocket TCP server."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            "0.0.0.0",
            self.port,
        )
        logger.info(f"[AudioSocket] Server listening on port {self.port}")

    async def stop(self):
        """Stop the AudioSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("[AudioSocket] Server stopped")

    async def _create_streaming_job_if_possible(self, session, audio_path: str):
        """Create DB job record for AudioSocket session when DB is available."""
        if self.staging_mode or self.db is None:
            return None

        from db_services.transcription import TranscriptionJobService

        transcription_svc = TranscriptionJobService(self.db)
        audio_size = None
        try:
            if audio_path and os.path.exists(audio_path):
                audio_size = os.path.getsize(audio_path)
        except OSError:
            audio_size = None

        await transcription_svc.create_job(
            job_id=session.job_id,
            audio_path=audio_path,
            is_async=True,
            enhance_audio=True,
            remove_silence=False,
            priority=0,
            callback_url=session.callback_url,
            audio_duration=session.duration,
            audio_size=audio_size,
        )
        return transcription_svc

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single AudioSocket connection (one phone call)."""
        peer = writer.get_extra_info("peername")
        peer_ip = peer[0] if isinstance(peer, tuple) and len(peer) > 0 else None
        logger.info(f"[AudioSocket] Connection from {peer}")

        if not is_ip_allowed(peer_ip, self.allowed_networks):
            logger.warning(f"[AudioSocket] Rejecting disallowed peer IP: {peer_ip}")
            writer.close()
            await writer.wait_closed()
            return

        if not self.streaming_asr:
            logger.error("[AudioSocket] Streaming ASR backend unavailable; rejecting connection")
            writer.close()
            await writer.wait_closed()
            return

        if not self.streaming_diar:
            logger.error("[AudioSocket] Streaming diarization backend unavailable; rejecting connection")
            writer.close()
            await writer.wait_closed()
            return

        session = None
        session_id = None
        latest_diar_result = {"segments": [], "num_speakers": 0}
        previous_model_text = ""
        cumulative_transcript = ""

        try:
            while True:
                if session and session.elapsed > self.max_session_duration_seconds:
                    logger.warning(f"[AudioSocket] Session {session_id} exceeded max duration")
                    break

                try:
                    header = await asyncio.wait_for(
                        reader.readexactly(HEADER_SIZE),
                        timeout=self.idle_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[AudioSocket] Idle timeout for session {session_id or 'unknown'}")
                    break
                except asyncio.IncompleteReadError:
                    logger.info(f"[AudioSocket] Connection closed: {session_id or 'unknown'}")
                    break

                frame_type = header[0]
                payload_len = struct.unpack("!H", header[1:3])[0]

                if payload_len > self.max_frame_bytes:
                    logger.warning(
                        f"[AudioSocket] Frame too large from {session_id or peer_ip}: {payload_len} bytes"
                    )
                    break

                payload = b""
                if payload_len > 0:
                    try:
                        payload = await asyncio.wait_for(
                            reader.readexactly(payload_len),
                            timeout=self.idle_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[AudioSocket] Payload read timeout for session {session_id or 'unknown'}"
                        )
                        break
                    except asyncio.IncompleteReadError:
                        logger.info(
                            f"[AudioSocket] Connection closed during payload read: {session_id or 'unknown'}"
                        )
                        break

                if session is None and frame_type != FRAME_UUID:
                    logger.warning(
                        f"[AudioSocket] First frame must be UUID. Closing connection from {peer_ip}"
                    )
                    break

                if frame_type == FRAME_UUID:
                    if session is not None:
                        logger.warning(
                            f"[AudioSocket] Duplicate UUID frame for active session {session_id}; closing connection"
                        )
                        break

                    if len(payload) >= 16:
                        call_uuid = str(uuid.UUID(bytes=payload[:16]))
                    else:
                        call_uuid = payload.hex()

                    session_id = call_uuid
                    logger.info(f"[AudioSocket] Call started: {session_id}")

                    try:
                        session = await self.session_manager.create_session(
                            session_id=session_id,
                            call_metadata={"source": "audiosocket", "peer": str(peer)},
                            callback_url=self.default_callback_url,
                            input_sample_rate=8000,
                        )
                    except (ValueError, RuntimeError) as exc:
                        logger.error(f"[AudioSocket] Failed to create session {session_id}: {exc}")
                        break

                elif frame_type == FRAME_AUDIO and session:
                    if not payload:
                        continue

                    session.add_audio(payload)
                    if session.current_ingress_bps() > self.max_bytes_per_second:
                        logger.warning(f"[AudioSocket] Ingress rate exceeded for session {session_id}")
                        break

                    chunk = session.get_chunk()
                    if chunk is None:
                        continue

                    chunk_offset = session.get_time_offset()
                    chunk_end = chunk_offset + (len(chunk) / 16000.0)

                    try:
                        asr_result, diar_result = await asyncio.wait_for(
                            asyncio.gather(
                                self.streaming_asr.transcribe_chunk(
                                    chunk,
                                    session_id,
                                    16000,
                                    request_id=f"{session_id}_asr_{session.chunks_processed}",
                                ),
                                self.streaming_diar.diarize_chunk(
                                    chunk,
                                    session_id,
                                    16000,
                                    request_id=f"{session_id}_diar_{session.chunks_processed}",
                                ),
                            ),
                            timeout=self.inference_timeout_seconds,
                        )

                        if diar_result.get("diar_ran"):
                            latest_diar_result = diar_result

                        # streaming_asr may emit sparse/non-cumulative text; merge robustly.
                        model_text = (asr_result.get("text") or "").strip()
                        if not model_text:
                            continue

                        delta_text, previous_model_text = extract_model_delta(
                            model_text,
                            previous_model_text,
                        )
                        if not delta_text:
                            continue

                        cumulative_transcript = append_transcript(cumulative_transcript, delta_text)
                        segment = build_chunk_segment(
                            delta_text,
                            latest_diar_result,
                            chunk_offset,
                            chunk_end,
                        )
                        if not segment:
                            continue

                        session.add_transcript_segment(segment)
                        transcript_msg = json.dumps(
                            {
                                "type": "transcript.update",
                                "is_final": False,
                                "update_mode": "sentence",
                                "cumulative_text": cumulative_transcript,
                                "segments": [segment],
                                "num_speakers": max(
                                    latest_diar_result.get("num_speakers", 0),
                                    segment.get("speaker_id", 0) + 1,
                                ),
                            }
                        ).encode("utf-8")
                        response_header = struct.pack(
                            "!BH",
                            FRAME_TRANSCRIPT_UPDATE,
                            len(transcript_msg),
                        )
                        writer.write(response_header + transcript_msg)
                        await writer.drain()
                    except asyncio.TimeoutError:
                        logger.error(f"[AudioSocket] Inference timeout for {session_id}")
                        break
                    except Exception as exc:
                        logger.error(f"[AudioSocket] Inference error for {session_id}: {exc}")

                elif frame_type == FRAME_SILENCE:
                    continue
                elif frame_type == FRAME_HANGUP:
                    logger.info(f"[AudioSocket] Hangup received: {session_id}")
                    break

        except Exception as exc:
            logger.error(f"[AudioSocket] Error: {exc}")
        finally:
            if session:
                try:
                    remaining = session.get_remaining_chunk()
                    final_segment = None
                    if remaining is not None:
                        remaining_offset = session.get_time_offset()
                        remaining_end = remaining_offset + (len(remaining) / 16000.0)
                        try:
                            asr_result, diar_result = await asyncio.wait_for(
                                asyncio.gather(
                                    self.streaming_asr.transcribe_chunk(
                                        remaining,
                                        session_id,
                                        16000,
                                        is_final=True,
                                    ),
                                    self.streaming_diar.diarize_chunk(
                                        remaining,
                                        session_id,
                                        16000,
                                        is_final=True,
                                    ),
                                ),
                                timeout=self.inference_timeout_seconds,
                            )
                            if diar_result.get("diar_ran"):
                                latest_diar_result = diar_result
                            final_text = (asr_result.get("text") or "").strip()
                            if final_text:
                                final_delta_text, previous_model_text = extract_model_delta(
                                    final_text,
                                    previous_model_text,
                                )
                                if final_delta_text:
                                    cumulative_transcript = append_transcript(
                                        cumulative_transcript,
                                        final_delta_text,
                                    )
                                    final_segment = build_chunk_segment(
                                        final_delta_text,
                                        latest_diar_result,
                                        remaining_offset,
                                        remaining_end,
                                    )
                                    if final_segment:
                                        session.add_transcript_segment(final_segment)
                        except Exception as exc:
                            logger.warning(f"[AudioSocket] Final inference failed for {session_id}: {exc}")

                    if final_segment:
                        try:
                            transcript_msg = json.dumps(
                                {
                                    "type": "transcript.update",
                                    "is_final": True,
                                    "update_mode": "sentence",
                                    "cumulative_text": cumulative_transcript,
                                    "segments": [final_segment],
                                    "num_speakers": max(
                                        latest_diar_result.get("num_speakers", 0),
                                        final_segment.get("speaker_id", 0) + 1,
                                    ),
                                }
                            ).encode("utf-8")
                            response_header = struct.pack(
                                "!BH",
                                FRAME_TRANSCRIPT_UPDATE,
                                len(transcript_msg),
                            )
                            writer.write(response_header + transcript_msg)
                            await writer.drain()
                        except Exception as exc:
                            logger.warning(
                                f"[AudioSocket] Could not send final transcript frame for {session_id}: {exc}"
                            )

                    audio_path = ""
                    try:
                        audio_path = session.finalize()
                    except Exception as exc:
                        logger.error(f"[AudioSocket] Failed to finalize audio for {session_id}: {exc}")

                    if audio_path:
                        try:
                            transcription_svc = await self._create_streaming_job_if_possible(
                                session,
                                audio_path,
                            )

                            if session.callback_url or transcription_svc:
                                from dependencies import process_audio_background_db

                                asyncio.create_task(
                                    process_audio_background_db(
                                        job_id=session.job_id,
                                        audio_path=audio_path,
                                        enhance_audio=True,
                                        remove_silence=False,
                                        priority=0,
                                        callback_url=session.callback_url,
                                        include_intelligence=True,
                                        intelligence_mode="full",
                                        api_key=None,
                                        transcription_svc=transcription_svc,
                                        call_metadata=session.call_metadata,
                                    )
                                )
                                logger.info(f"[AudioSocket] Post-call pipeline started: {session_id}")
                            else:
                                logger.warning(
                                    f"[AudioSocket] No callback_url and no DB service for {session_id}; "
                                    "skipping post-call pipeline"
                                )
                        except Exception as exc:
                            logger.error(f"[AudioSocket] Post-call pipeline error: {exc}")
                finally:
                    await self.session_manager.end_session(session.session_id)

            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
