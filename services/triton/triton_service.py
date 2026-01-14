
import asyncio
from intelligent_pipeline.pipeline import Pipeline

async def run_async_pipeline(audio_path: str, request_id: str, callback = None):
    """Convenience function to run the pipeline synchronously (e.g. for scripts)"""
    pipeline = Pipeline()
    raw_trans, labeled_trans, analysis, diar_info = await pipeline.run_pipeline(audio_path, request_id)
    if callback:
        callback(raw_trans, labeled_trans, analysis, diar_info)
