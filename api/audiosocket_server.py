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
import re
import struct
import uuid
from typing import Optional

from loguru import logger

from api.streaming_security import is_ip_allowed, parse_allowed_networks
from webhook import webhook_sender

# AudioSocket frame types
FRAME_HANGUP = 0x00
FRAME_UUID = 0x01
FRAME_SILENCE = 0x02
FRAME_AUDIO = 0x10
FRAME_TRANSCRIPT_UPDATE = 0x20
SENTENCE_SPLIT_RE = re.compile(r".+?(?:[.!?]+)(?=\s|$)")
FORCED_SENTENCE_WORDS = 14

HEADER_SIZE = 3  # 1 byte type + 2 bytes length


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

    if lower_new == lower_committed:
        return "", committed
    if lower_committed.endswith(lower_new):
        return "", committed
    if f" {lower_new} " in f" {lower_committed} ":
        return "", committed

    if lower_new.startswith(lower_committed):
        delta = new_text[len(committed):].strip()
        if not delta or not any(ch.isalnum() for ch in delta):
            return "", committed
        return delta, _append_live_text(committed, delta)

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
    """Build a single non-overlapping transcript segment for streaming output."""
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
        committed_transcript = ""
        emission_cursor = 0.0
        live_sentence_buffer = ""

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

                        emission_cursor = max(emission_cursor, session.get_time_offset())
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

                            transcript_msg = json.dumps(
                                {
                                    "type": "transcript.update",
                                    "is_final": False,
                                    "update_mode": "sentence",
                                    "cumulative_text": committed_transcript,
                                    "segments": emitted_segments,
                                    "num_speakers": max(
                                        latest_diar_result.get("num_speakers", 0),
                                        max(seg.get("speaker_id", 0) + 1 for seg in emitted_segments),
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
                    if remaining is not None:
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
                                final_delta_text, committed_transcript = _extract_incremental_delta(
                                    final_text,
                                    committed_transcript,
                                )
                                if final_delta_text:
                                    live_sentence_buffer = _append_live_text(
                                        live_sentence_buffer,
                                        final_delta_text,
                                    )
                        except Exception as exc:
                            logger.warning(f"[AudioSocket] Final inference failed for {session_id}: {exc}")

                    final_chunks, live_sentence_buffer = _extract_sentence_chunks(
                        live_sentence_buffer,
                        force_flush=True,
                    )
                    if final_chunks:
                        try:
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

                                transcript_msg = json.dumps(
                                    {
                                        "type": "transcript.update",
                                        "is_final": True,
                                        "update_mode": "sentence",
                                        "cumulative_text": committed_transcript,
                                        "segments": final_segments,
                                        "num_speakers": max(
                                            latest_diar_result.get("num_speakers", 0),
                                            max(seg.get("speaker_id", 0) + 1 for seg in final_segments),
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
