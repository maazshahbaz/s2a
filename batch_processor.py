import asyncio
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import time
from loguru import logger
from collections import deque
import threading
from enum import Enum
from chunking_utils import AudioChunk

class BatchStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BatchJob:
    job_id: str
    audio_data: np.ndarray
    metadata: Dict = field(default_factory=dict)
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    # Add chunking support
    audio_chunks: Optional[List[AudioChunk]] = None
    stitch_function: Optional[Callable] = None
    is_chunked: bool = False

@dataclass
class BatchConfig:
    max_batch_size: int = 128  # Increased for higher throughput
    max_queue_size: int = 500  # Increased for higher throughput
    processing_timeout: float = 300.0  # 5 minutes
    dynamic_batching: bool = True
    batch_timeout_ms: int = 100
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = True

class GPUMemoryManager:
    def __init__(self, target_utilization: float = 0.8):
        self.target_utilization = target_utilization
        self._lock = threading.Lock()
        
    def get_available_memory(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        
        with self._lock:
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            available = (total_memory - allocated_memory) / total_memory
            
        return available
    
    def can_process_batch(self, batch_size: int, estimated_memory_per_item: float = 0.1) -> bool:
        available = self.get_available_memory()
        required = batch_size * estimated_memory_per_item
        
        return available >= required
    
    def optimize_batch_size(self, requested_size: int, max_size: int) -> int:
        if not torch.cuda.is_available():
            return min(requested_size, max_size)
        
        available_memory = self.get_available_memory()
        
        # Estimate memory usage per item (rough approximation)
        memory_per_item = 0.1  # 10% of GPU memory per item (conservative)
        
        max_items_by_memory = int(available_memory * self.target_utilization / memory_per_item)
        optimal_size = min(requested_size, max_size, max_items_by_memory)
        
        return max(1, optimal_size)

class DynamicBatcher:
    def __init__(self, config: BatchConfig):
        self.config = config
        self.job_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.result_store: Dict[str, BatchJob] = {}
        self.processing_jobs: Dict[str, BatchJob] = {}
        self.gpu_manager = GPUMemoryManager(config.gpu_memory_fraction)
        self._stats = {
            'jobs_processed': 0,
            'total_processing_time': 0.0,
            'average_batch_size': 0.0,
            'gpu_utilization': 0.0
        }
        self._lock = asyncio.Lock()
        
    async def add_job(self, job: BatchJob) -> None:
        try:
            await asyncio.wait_for(
                self.job_queue.put(job), 
                timeout=1.0
            )
            logger.debug(f"Added job {job.job_id} to queue")
        except asyncio.TimeoutError:
            raise RuntimeError("Batch queue is full")
    
    async def get_result(self, job_id: str, timeout: float = 30.0) -> Optional[BatchJob]:
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with self._lock:
                if job_id in self.result_store:
                    return self.result_store.pop(job_id)
                
                if job_id in self.processing_jobs:
                    job = self.processing_jobs[job_id]
                    if job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
                        self.processing_jobs.pop(job_id)
                        return job
            
            await asyncio.sleep(0.1)
        
        return None
    
    def create_optimal_batches(self, jobs: List[BatchJob]) -> List[List[BatchJob]]:
        if not jobs:
            return []
        
        # Sort by priority and creation time
        jobs.sort(key=lambda x: (-x.priority, x.created_at))
        
        batches = []
        current_batch = []
        
        for job in jobs:
            # Check if adding this job would exceed limits
            if (len(current_batch) >= self.config.max_batch_size or
                (current_batch and not self._can_batch_together(current_batch[0], job))):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
            
            current_batch.append(job)
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _can_batch_together(self, job1: BatchJob, job2: BatchJob) -> bool:
        # Check if jobs can be batched together based on audio characteristics
        audio1_duration = len(job1.audio_data) / job1.metadata.get('sample_rate', 16000)
        audio2_duration = len(job2.audio_data) / job2.metadata.get('sample_rate', 16000)
        
        # Separate long audios (>24min) from short audios for optimal processing
        is_long_1 = audio1_duration > 24 * 60
        is_long_2 = audio2_duration > 24 * 60
        
        # Don't batch long audios together (they need individual chunking)
        if is_long_1 or is_long_2:
            return False
        
        # For short audios, don't batch if duration difference is too large
        max_duration = max(audio1_duration, audio2_duration)
        min_duration = min(audio1_duration, audio2_duration)
        
        if max_duration / min_duration > 3.0:  # Max 3x duration difference
            return False
        
        return True
    
    async def collect_batch(self, timeout_ms: int = 100) -> List[BatchJob]:
        """Collect exactly one job per worker for optimal parallelism"""
        try:
            job = await asyncio.wait_for(
                self.job_queue.get(), 
                timeout=timeout_ms / 1000.0
            )
            return [job]  # Return single job for 1-job-per-worker approach
        except asyncio.TimeoutError:
            return []  # No job available
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            'queue_size': self.job_queue.qsize(),
            'processing_jobs': len(self.processing_jobs),
            'gpu_memory_available': self.gpu_manager.get_available_memory()
        }

class BatchProcessor:
    def __init__(self, 
                 asr_service,
                 config: BatchConfig = None,
                 num_workers: int = 8):  # Increased for higher throughput
        
        self.asr_service = asr_service
        self.config = config or BatchConfig()
        self.num_workers = num_workers
        self.batcher = DynamicBatcher(self.config)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self._running = False
        self._worker_tasks = []
        
    async def start(self):
        if self._running:
            return
        
        self._running = True
        logger.info(f"Starting batch processor with {self.num_workers} workers")
        
        # Start worker tasks
        for i in range(self.num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(task)
    
    async def stop(self):
        self._running = False
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        self.executor.shutdown(wait=True)
        logger.info("Batch processor stopped")
    
    async def _worker_loop(self, worker_name: str):
        logger.info(f"Worker {worker_name} started")
        
        while self._running:
            try:
                # Collect exactly one job per worker
                jobs = await self.batcher.collect_batch(self.config.batch_timeout_ms)
                
                if not jobs:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process the single job (no batching optimization needed)
                await self._process_batch(jobs, worker_name)
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in worker {worker_name}: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_batch(self, jobs: List[BatchJob], worker_name: str):
        if not jobs:
            return
        
        # Should only be 1 job per call now
        job = jobs[0]
        job_start_time = time.time()
        
        logger.info(f"Worker {worker_name} processing job {job.job_id}")
        
        # Mark job as processing
        async with self.batcher._lock:
            job.status = BatchStatus.PROCESSING
            self.batcher.processing_jobs[job.job_id] = job
        
        try:
            # Check if audio needs chunking (>24 minutes)
            sr = job.metadata.get('sample_rate', 16000)
            duration = len(job.audio_data) / sr
            
            if duration > 24 * 60:  # 24 minutes
                logger.info(f"Job {job.job_id}: Long audio ({duration/60:.1f}min), using intelligent chunking")
                
                # Use ASR service's intelligent chunking
                audio_chunks, stitch_function = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.asr_service.chunk_audio_intelligent,
                    job.audio_data, sr
                )
                
                # Process chunks
                chunk_results = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.asr_service.transcribe_batch_nemo,
                    audio_chunks
                )
                
                # Stitch transcriptions using intelligent stitching
                final_result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    stitch_function,
                    chunk_results
                )
                
                processing_time = time.time() - job_start_time
                rtf = processing_time / duration if duration > 0 else float('inf')
                
                job.result = {
                    'text': final_result.get('text', ''),
                    'duration': duration,
                    'rtf': rtf,
                    'processing_time': processing_time,
                    'chunks_processed': final_result.get('chunks_processed', len(audio_chunks)),
                    'confidence': final_result.get('confidence')
                }
                job.is_chunked = True
                
            else:
                logger.info(f"Job {job.job_id}: Short audio ({duration/60:.1f}min), single chunk processing")
                
                # Process as single audio chunk
                results = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.asr_service.transcribe_batch_nemo,
                    [AudioChunk(
                        audio_data=job.audio_data,
                        start_time=0,
                        end_time=duration,
                        duration=duration,
                        chunk_id=0
                    )]
                )
                
                if results:
                    processing_time = time.time() - job_start_time
                    rtf = processing_time / duration if duration > 0 else float('inf')
                    
                    job.result = {
                        'text': results[0].get('text', ''),
                        'duration': duration,
                        'rtf': rtf,
                        'processing_time': processing_time,
                        'chunks_processed': 1,
                        'confidence': results[0].get('confidence')
                    }
                else:
                    job.result = {'text': '', 'duration': duration, 'rtf': float('inf')}
                    
                job.is_chunked = False
            
            job.status = BatchStatus.COMPLETED if job.result.get('text') else BatchStatus.FAILED
            if not job.result.get('text'):
                job.error = "No transcription generated"
                
            logger.info(f"Job {job.job_id} completed: {duration:.1f}s audio, "
                       f"RTF: {job.result.get('rtf', 0):.3f}, "
                       f"chunks: {job.result.get('chunks_processed', 1)}")
                
        except Exception as job_error:
            logger.error(f"Error processing job {job.job_id}: {job_error}")
            job.status = BatchStatus.FAILED
            job.error = str(job_error)
            job.result = {'text': '', 'duration': 0, 'rtf': float('inf')}
        
        # Move completed job to result store
        async with self.batcher._lock:
            self.batcher.result_store[job.job_id] = job
            if job.job_id in self.batcher.processing_jobs:
                del self.batcher.processing_jobs[job.job_id]
        
        # Update stats
        processing_time = time.time() - job_start_time
        self.batcher._stats['jobs_processed'] += 1
        self.batcher._stats['total_processing_time'] += processing_time
        self.batcher._stats['average_batch_size'] = 1.0  # Always 1 job per worker now
    
    async def transcribe_async(self, 
                             job_id: str,
                             audio_data: np.ndarray,
                             metadata: Dict = None,
                             priority: int = 0,
                             timeout: float = 300.0) -> Optional[Dict]:
        
        job = BatchJob(
            job_id=job_id,
            audio_data=audio_data,
            metadata=metadata or {},
            priority=priority
        )
        
        try:
            # Add job to queue
            await self.batcher.add_job(job)
            
            # Wait for result
            result_job = await self.batcher.get_result(job_id, timeout)
            
            if result_job is None:
                logger.warning(f"Job {job_id} timed out")
                return None
            
            if result_job.status == BatchStatus.FAILED:
                logger.error(f"Job {job_id} failed: {result_job.error}")
                return None
            
            return result_job.result
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        return self.batcher.get_stats()