from .intelligent_pipeline.audio_chunking import AudioChunking
from .intelligent_pipeline.transcription_client import AsyncTranscriptionService
from .intelligent_pipeline.diarization_client import AsyncDiarizationClient
from .intelligent_pipeline.analysis_client import AsyncAnalysis
from .intelligent_pipeline.diarization import GlobalDiarizationManager
from .intelligent_pipeline.transcript_merger import WordLevelDiarizationMerger

import asyncio
import os
import soundfile as sf
from typing import Tuple, Dict

class TritonService:
    """
    Complete pipeline that runs diarization on COMPLETE audio FIRST,
    then uses those global speaker labels for all chunks.
    
    This ensures consistent speaker IDs (e.g., exactly 2 speakers for a 2-person call).
    """
    
    def __init__(self, diarization_url: str = "host.docker.internal:2001"):
        self.chunking = AudioChunking()
        self.transcription = AsyncTranscriptionService(url=diarization_url)
        self.diarization = AsyncDiarizationClient(url=diarization_url)
        self.analysis = AsyncAnalysis(url=diarization_url)
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
    
    async def run_pipeline_async(self, audio_path: str, request_id) -> Tuple[str, str, str, Dict]:
        """
        Run complete pipeline with GLOBAL diarization.
        
        Key difference: Diarization runs on COMPLETE audio FIRST,
        then ASR runs on chunks, and results are merged using global speaker labels.
        
        Returns:
            (raw_transcription, labeled_transcription, analysis, diarization_info)
        """
        
        # Step 1: Run GLOBAL diarization on COMPLETE audio FIRST
        
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

        
        # Step 2: Create chunks for ASR (ASR may need smaller chunks)

        chunk_paths, chunk_timings = await self.chunking.create_chunks_async(audio_path)

        
        # Use full paths for transcription so Triton can find the files
        transcription_tasks = [
            self.transcription.transcribe_async(path, request_id)
            for path in chunk_paths
        ]
        
        transcriptions = await asyncio.gather(*transcription_tasks)
        
        raw_transcription, labeled_transcription, aligned_words = \
            self.merger.merge_all_chunks_with_global_diarization(
                transcriptions,
                self.global_diar_manager,
                chunk_timings
            )

        
        # Step 5: Analysis (using RAW transcription, NOT labeled)
        
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

def run_async_pipeline(audio_path: str, request_id: str, callback = None):
    """Convenience function to run the pipeline synchronously (e.g. for scripts)"""
    pipeline = TritonService()
    raw_trans, labeled_trans, analysis, diar_info = asyncio.run(pipeline.run_pipeline_async(audio_path, request_id))
    if callback:
        callback(raw_trans, labeled_trans, analysis, diar_info)
