"""
Worker for processing audio chunks in batches on GPU.
Pulls chunks from Redis queue and processes them with cross-job batching.
"""

import asyncio
import numpy as np
import soundfile as sf
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from db_services.transcription import TranscriptionJobService
from .chunk_metadata import ChunkMetadata, ChunkResult
from .redis_queue_manager import RedisQueueManager
from .stitching_service import StitchingService
from .chunking_utils import AudioChunk
from .alignment_service import align_sentence_segments, render_speaker_attributed_text
from .diarization_service import load_diar_segments
from config import get_diarization_settings


class AudioCache:
    """
    Cache audio files in memory to avoid repeated disk reads.
    Multiple chunks from same audio can reuse cached data.
    """

    def __init__(self, max_cache_size: int = 10):
        self._cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self._access_times: Dict[str, float] = {}
        self.max_cache_size = max_cache_size
        self._lock = asyncio.Lock()

    async def get_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Get audio from cache or load from disk"""
        async with self._lock:
            # Check cache first
            if audio_path in self._cache:
                self._access_times[audio_path] = time.time()
                return self._cache[audio_path]

            # Load from disk
            loop = asyncio.get_event_loop()
            audio, sr = await loop.run_in_executor(
                None, self._load_audio_sync, audio_path
            )

            # Add to cache (evict LRU if needed)
            if len(self._cache) >= self.max_cache_size:
                # Find least recently used
                lru_path = min(self._access_times.items(), key=lambda x: x[1])[0]
                del self._cache[lru_path]
                del self._access_times[lru_path]
                logger.debug(f"Evicted {lru_path} from audio cache")

            self._cache[audio_path] = (audio, sr)
            self._access_times[audio_path] = time.time()

            return audio, sr

    def _load_audio_sync(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio from disk (synchronous)"""
        audio, sr = sf.read(audio_path)
        # Convert to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr

    async def clear(self):
        """Clear the cache"""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()


class ChunkWorker:
    """
    Worker that processes chunks in batches.

    Key features:
    - Pulls chunks from Redis queue (FIFO)
    - Batches chunks from different jobs together
    - Processes up to batch_size chunks in one GPU batch
    - Uses in-memory audio extraction (no chunk files)
    - Automatically triggers stitching when job completes
    """

    def __init__(
        self,
        db,
        worker_id: str,
        asr_service,
        redis_queue: RedisQueueManager,
        batch_size: int = 128,
        audio_cache: Optional[AudioCache] = None,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        self.db=db
        self.worker_id = worker_id
        self.asr_service = asr_service
        self.redis_queue = redis_queue
        self.batch_size = batch_size
        self.audio_cache = audio_cache or AudioCache()
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        self._running = False
        self._stats = {
            'chunks_processed': 0,
            'batches_processed': 0,
            'total_audio_duration': 0,
            'total_processing_time': 0
        }

    async def start(self):
        """Start the worker processing loop"""
        self._running = True
        logger.info(f"Worker {self.worker_id} started (batch_size={self.batch_size})")

        try:
            while self._running:
                await self._process_batch()
                # Small delay between batches
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info(f"Worker {self.worker_id} cancelled")
        except Exception as e:
            logger.error(f"Worker {self.worker_id} crashed: {e}")
        finally:
            # Clean up any chunks still marked as processing
            await self.redis_queue.cleanup_worker_processing(self.worker_id)
            logger.info(f"Worker {self.worker_id} stopped")

    async def stop(self):
        """Stop the worker"""
        self._running = False

    async def _process_batch(self):
        """Process one batch of chunks"""
        # Pull chunks from queue
        chunks = await self.redis_queue.dequeue_chunks(
            worker_id=self.worker_id,
            batch_size=self.batch_size
        )

        if not chunks:
            return

        batch_start_time = time.time()

        try:
            # Extract audio segments in memory
            audio_segments = await self._extract_audio_segments(chunks)

            # Process on GPU
            results = await self._process_on_gpu(audio_segments)

            # Store results and check for job completion
            await self._store_results(chunks, results, batch_start_time)

            # Update stats
            self._update_stats(chunks, batch_start_time)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return chunks to queue on failure
            await self.redis_queue.cleanup_worker_processing(self.worker_id)

    async def _extract_audio_segments(
        self,
        chunks: List[ChunkMetadata]
    ) -> List[AudioChunk]:
        """
        Extract audio segments from full audio files.
        No physical chunk files - only in-memory extraction.
        """
        segments = []

        # Group chunks by audio path for efficient loading
        chunks_by_audio = {}
        for chunk in chunks:
            if chunk.audio_path not in chunks_by_audio:
                chunks_by_audio[chunk.audio_path] = []
            chunks_by_audio[chunk.audio_path].append(chunk)

        # Load each audio file once and extract all its chunks
        for audio_path, audio_chunks in chunks_by_audio.items():
            # Get audio from cache or disk
            audio, sr = await self.audio_cache.get_audio(audio_path)

            # Extract segments for this audio
            for chunk in audio_chunks:
                # Calculate sample indices
                start_sample = int(chunk.start_time * sr)
                end_sample = int(chunk.end_time * sr)

                # Extract segment (in-memory slicing)
                segment_audio = audio[start_sample:end_sample]

                # Create AudioChunk object for ASR processing
                audio_chunk = AudioChunk(
                    audio_data=segment_audio,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    duration=chunk.duration,
                    chunk_id=chunk.chunk_index  # Use index as ID for ordering
                )

                # Store metadata for result creation
                audio_chunk.metadata = chunk  # Attach original metadata

                segments.append(audio_chunk)

        logger.debug(
            f"Extracted {len(segments)} segments from {len(chunks_by_audio)} audio files"
        )

        return segments

    async def _process_on_gpu(
        self,
        audio_segments: List[AudioChunk]
    ) -> List[Dict]:
        """
        Process audio segments on GPU in a single batch.

        This is where the magic happens:
        - Batch can contain chunks from different users/jobs
        - GPU processes all chunks together for maximum efficiency
        - Can achieve RTFx of 3300 with batch_size=128
        """
        loop = asyncio.get_event_loop()

        # Process batch on GPU (blocking operation)
        results = await loop.run_in_executor(
            self.executor,
            self.asr_service.transcribe_batch_nemo,
            audio_segments
        )

        # Log performance
        total_duration = sum(seg.duration for seg in audio_segments)
        unique_jobs = len(set(seg.metadata.job_id for seg in audio_segments))

        logger.info(
            f"Worker {self.worker_id}: Processed batch of {len(audio_segments)} chunks "
            f"from {unique_jobs} jobs, total audio: {total_duration:.1f}s"
        )

        return results

    async def _store_results(
        self,
        chunks: List[ChunkMetadata],
        transcriptions: List[Dict],
        batch_start_time: float
    ):
        """Store results and trigger stitching if job is complete"""
        processing_time = time.time() - batch_start_time
        jobs_to_stitch = set()

        for chunk, transcription in zip(chunks, transcriptions):
            # Calculate per-chunk metrics
            chunk_processing_time = processing_time / len(chunks)
            rtf = chunk_processing_time / chunk.duration if chunk.duration > 0 else 0

            # Create result
            result = ChunkResult(
                chunk_id=chunk.chunk_id,
                job_id=chunk.job_id,
                chunk_index=chunk.chunk_index,
                text=transcription.get('text', ''),
                confidence=transcription.get('confidence', 0.0),
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                processing_time=chunk_processing_time,
                rtf=rtf,
                overlap_start=chunk.overlap_start,
                overlap_end=chunk.overlap_end
            )

            # Store result and check if job is complete
            job_complete = await self.redis_queue.mark_chunk_completed(
                self.worker_id,
                result
            )

            if job_complete:
                jobs_to_stitch.add(chunk.job_id)

        # Trigger stitching for completed jobs
        if jobs_to_stitch:
            for job_id in jobs_to_stitch:
                await self._trigger_stitching(job_id)

    async def _trigger_stitching(self, job_id: str):
        """Trigger stitching for a completed job and send webhook"""
        logger.info(f"Job {job_id} complete, triggering stitching")

        # Get all chunk results
        chunk_results = await self.redis_queue.get_job_results(job_id)

        if not chunk_results:
            logger.error(f"No results found for job {job_id}")
            return

        # Perform stitching (with config values from asr_service)
        stitching_service = StitchingService(
            words_per_second=self.asr_service.words_per_second,
            overlap_similarity_threshold=self.asr_service.overlap_similarity_threshold
        )

        transcription_svc = TranscriptionJobService(self.db)

        try:
            final_text = await stitching_service.stitch_transcriptions(chunk_results, remove_overlap=True)
        except Exception as e:
            logger.exception(f"Stitching failed for job {job_id}: {e}")
            await transcription_svc.update_job_status(job_id, 'failed', error_message=str(e))
            await self.redis_queue.update_job_status(job_id, 'error')
            return

        # Calculate overall metrics
        overall_confidence = stitching_service.calculate_confidence(chunk_results)
        overall_rtf = stitching_service.calculate_rtf(chunk_results)
        total_duration = sum((c.end_time - c.start_time) for c in chunk_results)
        total_processing_time = sum(c.processing_time for c in chunk_results)

        # Wait for diarization results with adaptive timeout based on audio duration
        # For 5-hour max audio: diarization takes ~75-90s, so we need adequate timeout
        def calculate_diarization_timeout(duration_seconds: float) -> int:
            """
            Calculate timeout iterations based on audio duration.
            Diarization speed: ~230x real-time (4s for 15min, 90s for 5hr)
            Max audio: 5 hours (18,000s)
            Max diarization time: ~90s
            """
            # Estimate diarization time (conservative: 200x real-time)
            estimated_diar_seconds = duration_seconds / 200.0
            
            # Add 50% buffer for safety
            timeout_seconds = estimated_diar_seconds * 1.5
            
            # Clamp between 6s (minimum) and 120s (maximum for 5hr audio)
            timeout_seconds = max(6.0, min(120.0, timeout_seconds))
            
            # Convert to iterations (each iteration = 0.1s)
            return int(timeout_seconds * 10)
        
        timeout_iterations = calculate_diarization_timeout(total_duration)
        logger.info(f"Waiting for diarization (timeout: {timeout_iterations/10:.1f}s for {total_duration:.1f}s audio)")
        
        diar_json = await load_diar_segments(self.redis_queue.redis, job_id)
        diarization_status = 'completed'
        
        if not diar_json:
            # Check if diarization failed
            diar_status_key = f"diar:{job_id}:status"
            diar_failure = await self.redis_queue.redis.get(diar_status_key)
            
            if diar_failure and diar_failure.decode() == 'failed':
                logger.warning(f"Diarization failed for job {job_id}, using single-speaker fallback")
                diarization_status = 'failed'
            else:
                # Poll with adaptive timeout
                for i in range(timeout_iterations):
                    await asyncio.sleep(0.1)
                    diar_json = await load_diar_segments(self.redis_queue.redis, job_id)
                    if diar_json:
                        logger.info(f"Diarization ready after {(i+1)/10:.1f}s wait")
                        break
                    
                    # Check for failure during polling
                    diar_failure = await self.redis_queue.redis.get(diar_status_key)
                    if diar_failure and diar_failure.decode() == 'failed':
                        logger.warning(f"Diarization failed during wait for job {job_id}")
                        diarization_status = 'failed'
                        break
                
                if not diar_json and diarization_status != 'failed':
                    logger.warning(f"Diarization timeout after {timeout_iterations/10:.1f}s for job {job_id}")
                    diarization_status = 'timeout'

        speaker_blocks = []
        num_speakers = 0

        # Align segments if diarization is available; else proceed with single-speaker
        if diar_json and diar_json.get('segments'):
            # Build ASR segments from chunk results
            asr_segments = []
            for c in chunk_results:
                if c.text:
                    asr_segments.append({
                        'start_time': c.start_time,
                        'end_time': c.end_time,
                        'text': c.text,
                    })
            
            logger.info(f"Aligning {len(asr_segments)} ASR segments with {len(diar_json['segments'])} diarization segments")
            logger.debug(f"Diarization speakers: {set(s['speaker'] for s in diar_json['segments'])}")
            
            speaker_blocks, num_speakers = align_sentence_segments(asr_segments, diar_json['segments'])
            
            logger.info(f"Alignment produced {len(speaker_blocks)} speaker blocks with {num_speakers} unique speakers")
            logger.debug(f"Speaker blocks: {[(b['speaker'], b['start'], b['end']) for b in speaker_blocks[:5]]}")
            
        else:
            # Fallback: assume single speaker
            speaker_blocks = [{
                'speaker': 'SPK_1',
                'start': min(c.start_time for c in chunk_results),
                'end': max(c.end_time for c in chunk_results),
                'text': final_text,
            }]
            num_speakers = 1

        # Update job status and save result with diarization in dedicated field
        await self.redis_queue.update_job_status(job_id, 'completed')
        diar_cfg = get_diarization_settings()
        diarization_data = {
            'speakerTranscript': speaker_blocks,
            'numSpeakers': num_speakers,
            'diarModel': diar_cfg.model_name,
            'diarizationStatus': diarization_status,  # 'completed', 'timeout', or 'failed'
            'audioDuration': total_duration
        }
        await transcription_svc.save_transcription_result(
            job_id,
            final_text,  # Plain text without speaker labels
            overall_confidence,
            overall_rtf,
            total_processing_time,
            len(chunk_results),
            diarization=diarization_data,  # Diarization data in dedicated field
        )

        # Get job metadata for webhook
        job_key = self.redis_queue.job_status_key(job_id)
        job_data = await self.redis_queue.redis.hgetall(job_key)

        if job_data:
            # Decode job data
            job_info = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in job_data.items()
            }

            callback_url = job_info.get('callback_url')

            if callback_url:
                # Prepare result for webhook
                result = {
                    'text': final_text,
                    'duration': total_duration,
                    'rtf': overall_rtf,
                    'processing_time': total_processing_time,
                    'chunks_processed': len(chunk_results),
                    'confidence': overall_confidence,
                    'diarization': diarization_data,
                }

                # Send webhook
                from webhook import webhook_sender, WebhookPayload
                webhook_payload = WebhookPayload(
                    job_id=job_id,
                    status="completed",
                    result=result,
                    processing_time=total_processing_time
                )

                # Send asynchronously
                asyncio.create_task(
                    webhook_sender.send_webhook(callback_url, webhook_payload)
                )
                logger.info(f"Job {job_id} completed, webhook sent to {callback_url}")

                # Save to database if available
                # Note: Database operations should be moved to a dedicated service
                # For now, we'll just log completion
                logger.info(f"Job {job_id} transcription complete: {len(final_text)} chars, RTF: {overall_rtf:.3f}")

    def _update_stats(self, chunks: List[ChunkMetadata], batch_time: float):
        """Update worker statistics"""
        self._stats['chunks_processed'] += len(chunks)
        self._stats['batches_processed'] += 1
        self._stats['total_audio_duration'] += sum(c.duration for c in chunks)
        self._stats['total_processing_time'] += batch_time

        # Log stats every 10 batches
        if self._stats['batches_processed'] % 10 == 0:
            avg_rtf = (
                self._stats['total_processing_time'] /
                self._stats['total_audio_duration']
                if self._stats['total_audio_duration'] > 0 else 0
            )
            logger.info(
                f"Worker {self.worker_id} stats: "
                f"{self._stats['chunks_processed']} chunks, "
                f"{self._stats['batches_processed']} batches, "
                f"avg RTF: {avg_rtf:.4f}"
            )

    def get_stats(self) -> Dict:
        """Get worker statistics"""
        return self._stats.copy()