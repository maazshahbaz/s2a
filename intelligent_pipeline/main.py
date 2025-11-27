import time
import asyncio

from pipeline import AsyncCompletePipelineWithGlobalDiarization

def test_callback(raw_trans, labeled_trans, analysis, diar_info):
    print(raw_trans)
    print(labeled_trans)
    print(analysis)
    print(diar_info)

def run_async_pipeline(audio_path: str, request_id: str, callback = None):
    """Convenience function to run the pipeline"""
    pipeline = AsyncCompletePipelineWithGlobalDiarization()
    raw_trans, labeled_trans, analysis, diar_info = asyncio.run(pipeline.run_pipeline_async(audio_path, request_id))
    callback(raw_trans, labeled_trans, analysis, diar_info)


if __name__ == "__main__":
    audio_path = "/home/sj/Desktop/data/back2/bytepulse-ai/uploads/2025-11-18/fe5cf860-62c3-45a0-8782-35fe0a482beb.wav"

    
    t1 = time.time()
    request_id = "1"
    run_async_pipeline(
        audio_path,
        request_id,
        test_callback
    )

    t2 = time.time()

    
    print(f"\n  Total Time: {t2 - t1:.2f} seconds")
