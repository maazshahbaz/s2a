import numpy as np
import time
import uuid
import os
import soundfile as sf
from scipy.signal import resample
from typing import Optional, Dict, List
from loguru import logger


class StreamingSession:
    """
    Manages state for a single streaming call session.

    Handles:
    - Audio buffering and chunking
    - 8kHz → 16kHz upsampling
    - Full audio accumulation for post-call processing
    - Session metadata and timing
    """

    TARGET_SR = 16000

    def __init__(
        self,
        session_id: str,
        call_metadata: Optional[Dict] = None,
        callback_url: Optional[str] = None,
        input_sample_rate: int = 8000,
        chunk_duration: float = 1.0,
    ):
        self.session_id = session_id
        self.job_id = str(uuid.uuid4())
        self.call_metadata = call_metadata or {}
        self.callback_url = callback_url
        self.input_sample_rate = input_sample_rate
        self.chunk_duration = chunk_duration

        # Audio buffers
        self._chunk_buffer = bytearray()  # PCM bytes waiting to be chunked
        self._total_audio = bytearray()   # All PCM bytes received (for post-call)

        # Tracking
        self.segment_counter = 0
        self.transcript_segments: List[Dict] = []
        self.start_time = time.time()
        self.total_samples_received = 0
        self.total_bytes_received = 0
        self.chunks_processed = 0

        # Chunk threshold: bytes needed for one chunk
        # PCM 16-bit = 2 bytes per sample
        self._chunk_threshold = int(
            input_sample_rate * chunk_duration * 2
        )

        logger.info(
            f"[Session {session_id}] Created: "
            f"input_sr={input_sample_rate}, chunk={chunk_duration}s, "
            f"threshold={self._chunk_threshold} bytes"
        )

    @property
    def duration(self) -> float:
        """Total audio duration received so far in seconds."""
        return self.total_samples_received / self.input_sample_rate

    @property
    def elapsed(self) -> float:
        """Wall-clock time since session started."""
        return time.time() - self.start_time

    def add_audio(self, pcm_bytes: bytes) -> None:
        """
        Add raw PCM audio bytes to the buffer.

        Args:
            pcm_bytes: Raw PCM audio (16-bit signed, little-endian)
        """
        self._chunk_buffer.extend(pcm_bytes)
        self._total_audio.extend(pcm_bytes)
        self.total_bytes_received += len(pcm_bytes)
        self.total_samples_received += len(pcm_bytes) // 2  # 16-bit = 2 bytes per sample

    def current_ingress_bps(self) -> float:
        """Average ingress rate in bytes/sec since session start."""
        elapsed = self.elapsed
        if elapsed < 2.0:
            return 0.0  # Grace period — don't enforce rate limit in first 2 seconds
        return self.total_bytes_received / elapsed

    def get_chunk(self) -> Optional[np.ndarray]:
        """
        If enough audio has accumulated, pop a chunk and return it as a 16kHz float32 array.

        Returns:
            np.ndarray (float32, 16kHz) or None if not enough audio yet.
        """
        if len(self._chunk_buffer) < self._chunk_threshold:
            return None

        # Pop chunk_threshold bytes from buffer
        chunk_bytes = bytes(self._chunk_buffer[: self._chunk_threshold])
        self._chunk_buffer = self._chunk_buffer[self._chunk_threshold:]

        # Convert PCM bytes to numpy array
        audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Upsample from input_sample_rate to 16kHz
        if self.input_sample_rate != self.TARGET_SR:
            num_target_samples = int(
                len(audio_float) * self.TARGET_SR / self.input_sample_rate
            )
            audio_float = resample(audio_float, num_target_samples).astype(np.float32)

        self.chunks_processed += 1
        return audio_float

    def get_remaining_chunk(self) -> Optional[np.ndarray]:
        """
        Get any remaining audio in the buffer (for the final chunk).
        Returns None if buffer is empty.
        """
        if len(self._chunk_buffer) < 2:  # Need at least 1 sample
            return None

        chunk_bytes = bytes(self._chunk_buffer)
        self._chunk_buffer.clear()

        # Ensure even number of bytes (16-bit samples)
        if len(chunk_bytes) % 2 != 0:
            chunk_bytes = chunk_bytes[:-1]

        audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Upsample
        if self.input_sample_rate != self.TARGET_SR:
            num_target_samples = int(
                len(audio_float) * self.TARGET_SR / self.input_sample_rate
            )
            audio_float = resample(audio_float, num_target_samples).astype(np.float32)

        self.chunks_processed += 1
        return audio_float

    def add_transcript_segment(self, segment: Dict) -> None:
        """Store a transcript segment for the full call transcript."""
        self.transcript_segments.append(segment)
        self.segment_counter += 1

    def get_time_offset(self) -> float:
        """Get the current time offset in seconds (how much audio has been processed)."""
        processed_samples = (self.chunks_processed - 1) * (
            self.input_sample_rate * self.chunk_duration
        )
        return max(0, processed_samples / self.input_sample_rate)

    def finalize(self, upload_dir: str = "/data/uploads") -> str:
        """
        Save the complete audio as a WAV file for post-call processing.

        Args:
            upload_dir: Directory to save the WAV file.

        Returns:
            Path to the saved WAV file.
        """
        audio_bytes = bytes(self._total_audio)

        # Ensure even number of bytes
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]

        if len(audio_bytes) < 2:
            logger.warning(f"[Session {self.session_id}] No audio to save")
            return ""

        # Convert to float32
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Upsample to 16kHz for the saved file
        if self.input_sample_rate != self.TARGET_SR:
            num_target_samples = int(
                len(audio_float) * self.TARGET_SR / self.input_sample_rate
            )
            audio_float = resample(audio_float, num_target_samples).astype(np.float32)

        # Save as WAV
        os.makedirs(upload_dir, exist_ok=True)
        audio_path = os.path.join(upload_dir, f"{self.job_id}.wav")
        sf.write(audio_path, audio_float, self.TARGET_SR)

        duration = len(audio_float) / self.TARGET_SR
        logger.info(
            f"[Session {self.session_id}] Saved {duration:.1f}s audio to {audio_path}"
        )

        # Clear buffers
        self._chunk_buffer.clear()
        self._total_audio.clear()

        return audio_path
