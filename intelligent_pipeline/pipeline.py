import asyncio
import os
import soundfile as sf
from typing import Tuple, Dict, List
from .config_loader import config

from audio_chunking import AudioChunking
from transcription_client import AsyncTranscriptionService
from diarization_client import AsyncDiarizationClient
from analysis_client import AsyncAnalysis
from transcript_merger import TranscriptMerger


class Pipeline:
    """
    Audio processing pipeline with GLOBAL speaker consistency.
    
    Key improvement: Aligns speaker labels across chunks using embedding similarity.
    
    Steps:
    1. Chunk audio to 5-minute intervals at 16 KHz
    2a. Run diarization on FULL audio (for global reference)
    2b. Run ASR on chunks (parallel)
    3. Align chunk-based transcription with global diarization
    4. Run AI analysis on transcription
    """
    
    def __init__(self, 
        diarization_url: str = None,
        transcription_url: str = None):
        """
        Initialize pipeline with service URLs from config or provided values.
        
        Args:
            diarization_url: Override diarization service URL (default: from config)
            transcription_url: Override transcription service URL (default: from config)
        """
        # Load service configurations
        diar_config = config.get_service_config('diarization')
        trans_config = config.get_service_config('transcription')
        
        self.chunking = AudioChunking()
        self.transcription = AsyncTranscriptionService(
            url=transcription_url or trans_config.get('url')
        )
        self.diarization = AsyncDiarizationClient(
            url=diarization_url or diar_config.get('url')
        )
        self.analysis = AsyncAnalysis()
        self.merger = TranscriptMerger()
    
    async def _delete_chunks_async(self, chunk_paths: List[str]):
        """Delete all chunk files asynchronously."""
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, os.remove, path) for path in chunk_paths]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        audio_data, sample_rate = sf.read(audio_path)
        return len(audio_data) / sample_rate
    
    async def _run_global_diarization(self, audio_path: str, request_id: str) -> Dict:
        """
        Step 2a: Run diarization on FULL audio file for global speaker consistency.
        
        This provides a reference for consistent speaker labels across all chunks.
        """
        await self.diarization.connect()
        
        global_diar_result = await self.diarization.diarize(audio_path, request_id)
        
        return global_diar_result
    
    async def _run_transcription_on_chunks(self, chunk_paths: List[str], request_id: str) -> List[Dict]:
        """Step 2b: Run transcription on all chunks in parallel."""
        chunk_basenames = [os.path.basename(path) for path in chunk_paths]
        
        transcription_tasks = [
            self.transcription.transcribe_async(basename, f"{request_id}_chunk_{i}")
            for i, basename in enumerate(chunk_basenames)
        ]
        
        transcription_results = await asyncio.gather(*transcription_tasks)
        return transcription_results
    
    def _align_segments_to_global(
        self,
        global_segments: List[Dict],
        chunk_timings: List[Tuple[float, float]]
    ) -> List[List[Dict]]:
        """
        Split global diarization segments into per-chunk segments.
        
        Args:
            global_segments: Segments from full audio diarization
            chunk_timings: [(start, end), ...] for each chunk
            
        Returns:
            List of segment lists, one per chunk
        """
        chunk_segments = []
        
        for chunk_start, chunk_end in chunk_timings:
            chunk_segs = []
            
            for seg in global_segments:
                seg_start = seg['start']
                seg_end = seg['end']
                
                # Check if segment overlaps with this chunk
                if seg_end <= chunk_start or seg_start >= chunk_end:
                    continue  # No overlap
                
                # Clip segment to chunk boundaries and adjust to chunk-local time
                local_start = max(0, seg_start - chunk_start)
                local_end = min(chunk_end - chunk_start, seg_end - chunk_start)
                
                chunk_segs.append({
                    'speaker': seg['speaker'],
                    'start': local_start,
                    'end': local_end,
                    'duration': local_end - local_start
                })
            
            chunk_segments.append(chunk_segs)
        
        return chunk_segments
    
    async def run_pipeline(self, audio_path: str, request_id: str) -> Tuple[str, str, str, Dict]:
        """
        Execute the complete pipeline with global speaker consistency.
        
        Args:
            audio_path: Path to input audio file
            request_id: Unique identifier for this request
            
        Returns:
            Tuple of (raw_transcription, labeled_transcription, analysis, metadata)
        """
        
        # Step 1: Chunk audio to 5-minute intervals
        chunk_paths, chunk_timings = await self.chunking.create_chunks_async(audio_path)
        
        # Step 2: Run global diarization and chunk transcription in parallel
        global_diar_task = self._run_global_diarization(audio_path, request_id)
        transcription_task = self._run_transcription_on_chunks(chunk_paths, request_id)
        
        global_diar_result, transcription_results = await asyncio.gather(
            global_diar_task,
            transcription_task
        )
        # Extract global segments
        global_segments = global_diar_result.get('segments', [])
        print(f"[Pipeline] Global diarization: {len(global_segments)} segments, "
              f"{global_diar_result.get('num_speakers', 0)} speakers")
        
        # Step 3: Align global segments to chunk boundaries
        chunk_diarization = self._align_segments_to_global(global_segments, chunk_timings)
        # Step 4: Merge transcriptions with globally-consistent diarization
        raw_transcription, labeled_transcription = await self.merger.merge_transcriptions(
            request_id,transcription_results,
            [{'segments': segs} for segs in chunk_diarization],
            chunk_timings
        )
        # Step 5: Run AI analysis and cleanup chunks in parallel
        analysis_task = self.analysis.analyze_call_async(labeled_transcription, request_id)
        cleanup_task = self._delete_chunks_async(chunk_paths)
        
        analysis, _ = await asyncio.gather(analysis_task, cleanup_task)
        
        # Close connections
        await self.diarization.close()
        
        # Prepare metadata
        metadata = {
            'chunk_count': len(chunk_paths),
            'chunk_timings': chunk_timings,
            'num_speakers': global_diar_result.get('num_speakers', 0),
            'total_segments': len(global_segments)
        }
        
        return raw_transcription, labeled_transcription, analysis, metadata