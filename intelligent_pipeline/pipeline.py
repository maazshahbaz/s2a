from audio_chunking import AudioChunking
from transcription_client import AsyncTranscriptionService
from diarization_client import AsyncDiarizationClient
from analysis_client import AsyncAnalysis
from diarization import GlobalDiarizationManager
from transcript_merger import WordLevelDiarizationMerger

import asyncio
import os
import soundfile as sf
from typing import Tuple, Dict

class AsyncCompletePipelineWithGlobalDiarization:
    """
    Complete pipeline that runs diarization on COMPLETE audio FIRST,
    then uses those global speaker labels for all chunks.
    
    This ensures consistent speaker IDs (e.g., exactly 2 speakers for a 2-person call).
    """
    
    def __init__(self, diarization_url: str = "localhost:2001"):
        self.chunking = AudioChunking()
        self.transcription = AsyncTranscriptionService()
        self.diarization = AsyncDiarizationClient(url=diarization_url)
        self.analysis = AsyncAnalysis()
        self.global_diar_manager = GlobalDiarizationManager()
        self.merger = WordLevelDiarizationMerger()
        
    async def __delete_chunks_async(self, chunk_paths):
        """Delete all chunks asynchronously."""
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, os.remove, path) for path in chunk_paths]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        audio_data, sample_rate = sf.read(audio_path)
        return len(audio_data) / sample_rate
    
    async def _run_global_diarization(self, audio_path: str, request_id) -> Dict:
        """
        Step 1: Run GLOBAL diarization on COMPLETE audio.
        Returns diarization summary.
        """
        await self.diarization.connect()
        
        # Get audio duration
        audio_duration = self._get_audio_duration(audio_path)
        
        # Diarize the COMPLETE audio file
        global_diar_result = await self.diarization.diarize(audio_path, request_id)
        
        # Store in global manager
        diar_summary = self.global_diar_manager.set_global_diarization(
            global_diar_result, 
            audio_duration
        )
        
        return diar_summary
    
    async def _run_chunking_and_transcription(self, audio_path: str, request_id) -> Tuple:
        """
        Step 2: Create chunks and transcribe them.
        Returns (transcriptions, chunk_paths, chunk_timings).
        """
        # Create chunks for ASR
        chunk_paths, chunk_timings = await self.chunking.create_chunks_async(audio_path)
        
        # Get basenames for transcription
        chunk_basenames = [os.path.basename(path) for path in chunk_paths]
        
        # Transcribe all chunks in parallel
        transcription_tasks = [
            self.transcription.transcribe_async(basename, request_id)
            for basename in chunk_basenames
        ]
        
        transcriptions = await asyncio.gather(*transcription_tasks)
        
        return transcriptions, chunk_paths, chunk_timings
    
    async def run_pipeline_async(self, audio_path: str, request_id) -> Tuple[str, str, str, Dict]:
        """
        Run complete pipeline with GLOBAL diarization.
        
        Key difference: Diarization and transcription run IN PARALLEL,
        then results are merged using global speaker labels.
        
        Returns:
            (raw_transcription, labeled_transcription, analysis, diarization_info)
        """
        
        # Steps 1 and 2: Run diarization and transcription IN PARALLEL
        diar_task = self._run_global_diarization(audio_path, request_id)
        transcription_task = self._run_chunking_and_transcription(audio_path, request_id)
        
        diar_summary, (transcriptions, chunk_paths, chunk_timings) = await asyncio.gather(
            diar_task,
            transcription_task
        )
        
        # Step 3: Merge transcriptions with global diarization
        raw_transcription, labeled_transcription, aligned_words = \
            self.merger.merge_all_chunks_with_global_diarization(
                transcriptions,
                self.global_diar_manager,
                chunk_timings
            )
        
        # Step 4: Analysis (using RAW transcription, NOT labeled)
        # CRITICAL: Pass raw_transcription to analysis, NOT labeled_transcription
        analysis_task = self.analysis.analyze_call_async(raw_transcription, request_id)
        delete_task = self.__delete_chunks_async(chunk_paths)
        
        analysis, _ = await asyncio.gather(analysis_task, delete_task)
        
        await self.diarization.close()
        
        # Prepare diarization info
        diarization_info = self.global_diar_manager.get_summary()
        diarization_info['aligned_segments_count'] = len(aligned_words)
        diarization_info['chunk_count'] = len(chunk_paths)
        diarization_info['chunk_timings'] = chunk_timings
        
        return raw_transcription, labeled_transcription, analysis, diarization_info