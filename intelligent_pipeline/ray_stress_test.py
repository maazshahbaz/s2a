import os
import time
import ray
import asyncio
import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from pydub import AudioSegment
import statistics

from pipeline import AsyncCompletePipelineWithGlobalDiarization


class AudioSplitter:
    """Handles audio file splitting into different durations"""
    
    def __init__(self, original_audio_path: str, output_dir: str = "./split_audio"):
        self.original_audio_path = original_audio_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Duration splits in minutes
        self.durations = [10, 20, 30, "full"]
        self.split_files = {}
        
    def split_audio(self) -> Dict[str, str]:
        """Split audio into different durations"""
        print(f"Loading audio file: {self.original_audio_path}")
        
        # Load the audio file
        audio = AudioSegment.from_file(self.original_audio_path)
        total_duration_ms = len(audio)
        total_duration_min = total_duration_ms / (1000 * 60)
        
        print(f"Total audio duration: {total_duration_min:.2f} minutes")
        
        # Create splits
        for duration in self.durations:
            if duration == "full":
                # Use the full audio
                output_path = self.output_dir / f"audio_full.wav"
                if not output_path.exists():
                    audio.export(str(output_path), format="wav")
                    print(f"Created full audio: {output_path}")
                else:
                    print(f"Full audio already exists: {output_path}")
                self.split_files["full"] = str(output_path)
            else:
                # Create split of specific duration
                duration_ms = duration * 60 * 1000
                
                if duration_ms > total_duration_ms:
                    print(f"Warning: Requested {duration}min but audio is only {total_duration_min:.2f}min. Using full audio.")
                    output_path = self.output_dir / f"audio_{duration}min.wav"
                    if not output_path.exists():
                        audio.export(str(output_path), format="wav")
                    self.split_files[f"{duration}min"] = str(output_path)
                else:
                    output_path = self.output_dir / f"audio_{duration}min.wav"
                    if not output_path.exists():
                        audio_segment = audio[:duration_ms]
                        audio_segment.export(str(output_path), format="wav")
                        print(f"Created {duration}min split: {output_path}")
                    else:
                        print(f"{duration}min split already exists: {output_path}")
                    self.split_files[f"{duration}min"] = str(output_path)
        
        return self.split_files
    
    def get_random_audio(self) -> Tuple[str, str]:
        """Get a random audio file from the splits"""
        duration_key = random.choice(list(self.split_files.keys()))
        return duration_key, self.split_files[duration_key]
    
    def create_audio_copy(self, request_id: str) -> Tuple[str, str]:
        """
        Create a copy of a random audio file with the request ID in the filename
        
        Args:
            request_id: Unique identifier for the request
            
        Returns:
            Tuple of (duration_label, path_to_copied_file)
        """
        import shutil
        
        # Get a random audio file
        duration_key = random.choice(list(self.split_files.keys()))
        original_path = self.split_files[duration_key]
        
        # Create a copy with request ID in the filename
        original_file = Path(original_path)
        file_extension = original_file.suffix
        copy_filename = f"{original_file.stem}_{request_id}{file_extension}"
        copy_path = self.output_dir / copy_filename
        
        # Copy the file
        shutil.copy2(original_path, copy_path)
        
        return duration_key, str(copy_path)


@ray.remote
class PipelineWorker:
    """Ray actor for processing pipeline requests"""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.pipeline = AsyncCompletePipelineWithGlobalDiarization()
        
    async def process_request(self, audio_path: str, request_id: str, duration_label: str):
        """Process a single request - async method that Ray can handle"""
        start_time = time.time()
        
        try:
            raw_trans, labeled_trans, analysis, diar_info = await self.pipeline.run_pipeline_async(
                audio_path, 
                request_id
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            return {
                "success": True,
                "request_id": request_id,
                "worker_id": self.worker_id,
                "duration_label": duration_label,
                "processing_time": processing_time,
                "raw_trans_length": len(raw_trans) if raw_trans else 0,
                "labeled_trans_length": len(labeled_trans) if labeled_trans else 0,
                "has_analysis": analysis is not None,
                "has_diar_info": diar_info is not None,
                "error": None
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            return {
                "success": False,
                "request_id": request_id,
                "worker_id": self.worker_id,
                "duration_label": duration_label,
                "processing_time": processing_time,
                "error": str(e)
            }


class StressTester:
    """Main stress testing orchestrator"""
    
    def __init__(self, original_audio_path: str, num_workers: int = 4):
        self.original_audio_path = original_audio_path
        self.num_workers = num_workers
        self.audio_splitter = AudioSplitter(original_audio_path)
        self.results = []
        self.copied_files = []  # Track copied files for cleanup
        
    def setup(self):
        """Initialize Ray and prepare audio splits"""
        print("Initializing Ray...")
        ray.init(ignore_reinit_error=True)
        
        print("\nSplitting audio files...")
        self.audio_splitter.split_audio()
        
        print(f"\nCreating {self.num_workers} worker actors...")
        self.workers = [PipelineWorker.remote(i) for i in range(self.num_workers)]
        
    def run_stress_test(self, num_requests: int, concurrent_requests: int = None):
        """
        Run stress test with specified parameters
        
        Args:
            num_requests: Total number of requests to send
            concurrent_requests: Number of concurrent requests (default: num_workers)
        """
        if concurrent_requests is None:
            concurrent_requests = self.num_workers
            
        print(f"\n{'='*80}")
        print(f"STRESS TEST CONFIGURATION")
        print(f"{'='*80}")
        print(f"Total requests: {num_requests}")
        print(f"Concurrent requests: {concurrent_requests}")
        print(f"Number of workers: {self.num_workers}")
        print(f"Audio variations: {list(self.audio_splitter.split_files.keys())}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        pending_requests = []
        completed = 0
        
        # Submit initial batch of requests
        for i in range(min(concurrent_requests, num_requests)):
            request_id = f"req_{i:04d}"
            duration_label, audio_path = self.audio_splitter.create_audio_copy(request_id)
            self.copied_files.append(audio_path)  # Track for cleanup
            worker = self.workers[i % self.num_workers]
            
            future = worker.process_request.remote(audio_path, request_id, duration_label)
            pending_requests.append((future, i, duration_label))
            print(f"Submitted request {i+1}/{num_requests} - {duration_label} - {request_id}")
        
        next_request_id = concurrent_requests
        
        # Process requests as they complete
        while pending_requests:
            # Wait for at least one request to complete
            ready, pending_futures = ray.wait([f[0] for f in pending_requests], num_returns=1, timeout=None)
            
            for ready_future in ready:
                # Find the completed request
                for idx, (future, req_idx, dur_label) in enumerate(pending_requests):
                    if future == ready_future:
                        result = ray.get(ready_future)
                        self.results.append(result)
                        completed += 1
                        
                        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
                        print(f"{status} - Request {req_idx+1} ({result['request_id']}) - "
                              f"{dur_label} - {result['processing_time']:.2f}s - "
                              f"Worker {result['worker_id']}")
                        
                        if not result["success"]:
                            print(f"  Error: {result['error']}")
                        
                        # Remove from pending
                        pending_requests.pop(idx)
                        
                        # Submit next request if available
                        if next_request_id < num_requests:
                            request_id = f"req_{next_request_id:04d}"
                            duration_label, audio_path = self.audio_splitter.create_audio_copy(request_id)
                            self.copied_files.append(audio_path)  # Track for cleanup
                            worker = self.workers[next_request_id % self.num_workers]
                            
                            new_future = worker.process_request.remote(audio_path, request_id, duration_label)
                            pending_requests.append((new_future, next_request_id, duration_label))
                            print(f"Submitted request {next_request_id+1}/{num_requests} - {duration_label} - {request_id}")
                            next_request_id += 1
                        
                        break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        self.print_statistics(total_time)
        
    def print_statistics(self, total_time: float):
        """Print detailed statistics about the stress test"""
        print(f"\n{'='*80}")
        print(f"STRESS TEST RESULTS")
        print(f"{'='*80}\n")
        
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        print(f"Total Requests: {len(self.results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Throughput: {len(self.results)/total_time:.2f} requests/second")
        
        if successful:
            processing_times = [r["processing_time"] for r in successful]
            print(f"\nProcessing Time Statistics:")
            print(f"  Mean: {statistics.mean(processing_times):.2f}s")
            print(f"  Median: {statistics.median(processing_times):.2f}s")
            print(f"  Min: {min(processing_times):.2f}s")
            print(f"  Max: {max(processing_times):.2f}s")
            print(f"  Std Dev: {statistics.stdev(processing_times):.2f}s" if len(processing_times) > 1 else "  Std Dev: N/A")
            
            # Statistics by duration
            print(f"\nStatistics by Audio Duration:")
            for duration in ["10min", "20min", "30min", "full"]:
                duration_results = [r for r in successful if r["duration_label"] == duration]
                if duration_results:
                    duration_times = [r["processing_time"] for r in duration_results]
                    print(f"  {duration}:")
                    print(f"    Count: {len(duration_results)}")
                    print(f"    Mean Time: {statistics.mean(duration_times):.2f}s")
                    print(f"    Min/Max: {min(duration_times):.2f}s / {max(duration_times):.2f}s")
            
            # Worker statistics
            print(f"\nStatistics by Worker:")
            for worker_id in range(self.num_workers):
                worker_results = [r for r in successful if r["worker_id"] == worker_id]
                if worker_results:
                    worker_times = [r["processing_time"] for r in worker_results]
                    print(f"  Worker {worker_id}:")
                    print(f"    Requests: {len(worker_results)}")
                    print(f"    Mean Time: {statistics.mean(worker_times):.2f}s")
        
        if failed:
            print(f"\nFailed Requests:")
            for r in failed:
                print(f"  {r['request_id']} ({r['duration_label']}): {r['error']}")
        
        print(f"\n{'='*80}\n")
    
    def cleanup(self):
        """Cleanup resources"""
        print("Shutting down Ray...")
        ray.shutdown()
        
        # Clean up copied audio files
        print(f"Cleaning up {len(self.copied_files)} copied audio files...")
        import os
        cleaned_count = 0
        for file_path in self.copied_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    cleaned_count += 1
            except Exception as e:
                print(f"  Warning: Failed to remove {file_path}: {e}")
        
        print(f"✓ Cleaned up {cleaned_count}/{len(self.copied_files)} copied files")


def main():
    """Main entry point for stress testing"""
    
    # Configuration
    ORIGINAL_AUDIO_PATH = "/home/sj/Desktop/data/back2/bytepulse-ai/uploads/2025-11-18/2a0c5881-6982-469e-a59b-1a9d8469870f.wav"
    NUM_WORKERS = 20 # Number of Ray workers
    NUM_REQUESTS = 20  # Total number of requests to send
    CONCURRENT_REQUESTS = 20  # Number of concurrent requests
    
    # Initialize stress tester
    tester = StressTester(
        original_audio_path=ORIGINAL_AUDIO_PATH,
        num_workers=NUM_WORKERS
    )
    
    try:
        # Setup
        tester.setup()
        
        # Run stress test
        tester.run_stress_test(
            num_requests=NUM_REQUESTS,
            concurrent_requests=CONCURRENT_REQUESTS
        )
        
    finally:
        # Cleanup
        tester.cleanup()


if __name__ == "__main__":
    main()