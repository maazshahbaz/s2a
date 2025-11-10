"""
Generate chunk metadata for audio files.
Does NOT create physical chunk files - only metadata for in-memory extraction.
"""

import numpy as np
from typing import List
from loguru import logger
from .chunk_metadata import ChunkMetadata


class ChunkGenerator:
    """
    Generate chunk metadata for audio files.

    Key features:
    - No physical chunk files created
    - Supports variable chunk sizes (1 second to 24 minutes)
    - Adds overlap for accurate stitching
    - Overlap is removed during stitching, not processing
    """

    @staticmethod
    def generate_chunks(
        job_id: str,
        audio_path: str,
        audio_duration: float,
        sample_rate: int = 16000,
        max_chunk_duration: float = 1440.0,  # 24 minutes (NVIDIA Parakeet limit)
        overlap_duration: float = 5.0,  # 5 seconds overlap
        include_intelligence: bool = False,
        callback_url: str = None
    ) -> List[ChunkMetadata]:
        """
        Generate chunk metadata for an audio file.

        Args:
            job_id: Unique job identifier
            audio_path: Path to the original audio file (stored once)
            audio_duration: Total duration of audio in seconds
            sample_rate: Audio sample rate
            max_chunk_duration: Maximum chunk duration (24 min for Parakeet)
            overlap_duration: Overlap between chunks for stitching
            callback_url: Webhook URL for completion notification

        Returns:
            List of ChunkMetadata objects (no actual audio data)
        """
        chunks = []

        # For short audio (≤24 minutes), create single chunk
        if audio_duration <= max_chunk_duration:
            chunk = ChunkMetadata(
                chunk_id=f"{job_id}_chunk_0",
                job_id=job_id,
                audio_path=audio_path,
                start_time=0.0,
                end_time=audio_duration,
                duration=audio_duration,
                chunk_index=0,
                total_chunks=1,
                sample_rate=sample_rate,
                callback_url=callback_url,
                overlap_start=0,
                overlap_end=0,
                include_intelligence=include_intelligence
            )
            chunks.append(chunk)

            logger.info(f"Job {job_id}: Single chunk for {audio_duration:.1f}s audio")

        else:
            # For long audio (>24 minutes), create multiple chunks with overlap

            # Calculate number of chunks needed
            # Each chunk is max_chunk_duration, with overlap_duration overlap
            effective_chunk_duration = max_chunk_duration - overlap_duration
            num_chunks = int(np.ceil((audio_duration - overlap_duration) / effective_chunk_duration))

            for i in range(num_chunks):
                # Calculate start and end times
                if i == 0:
                    # First chunk: no overlap at start
                    start = 0.0
                    end = min(max_chunk_duration, audio_duration)
                    overlap_start = 0.0
                    overlap_end = overlap_duration if i < num_chunks - 1 else 0.0

                elif i == num_chunks - 1:
                    # Last chunk: no overlap at end
                    start = i * effective_chunk_duration
                    end = audio_duration
                    overlap_start = overlap_duration
                    overlap_end = 0.0

                else:
                    # Middle chunks: overlap on both sides
                    start = i * effective_chunk_duration
                    end = min(start + max_chunk_duration, audio_duration)
                    overlap_start = overlap_duration
                    overlap_end = overlap_duration

                # Create chunk metadata
                chunk = ChunkMetadata(
                    chunk_id=f"{job_id}_chunk_{i}",
                    job_id=job_id,
                    audio_path=audio_path,
                    start_time=start,
                    end_time=end,
                    duration=end - start,
                    chunk_index=i,
                    total_chunks=num_chunks,
                    sample_rate=sample_rate,
                    callback_url=callback_url,
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                    include_intelligence=include_intelligence
                )
                chunks.append(chunk)

            logger.info(
                f"Job {job_id}: Generated {num_chunks} chunks for {audio_duration:.1f}s audio "
                f"(chunk_size: {max_chunk_duration}s, overlap: {overlap_duration}s)"
            )

        return chunks

    @staticmethod
    def estimate_processing_time(
        chunks: List[ChunkMetadata],
        target_rtf: float = 0.03  # RTFx 3300 = RTF 0.0003, but use conservative estimate
    ) -> float:
        """
        Estimate processing time for chunks.

        With batch_size=128 and RTFx of 3300:
        - Processing 128 chunks of 24 minutes = 3072 minutes of audio
        - Time = 3072 minutes / 3300 = ~56 seconds

        Args:
            chunks: List of chunk metadata
            target_rtf: Target Real-Time Factor (0.03 = RTFx 33)

        Returns:
            Estimated processing time in seconds
        """
        total_duration = sum(c.duration for c in chunks)
        estimated_time = total_duration * target_rtf

        logger.debug(
            f"Estimated processing time: {estimated_time:.1f}s for "
            f"{total_duration:.1f}s audio (RTF: {target_rtf})"
        )

        return estimated_time