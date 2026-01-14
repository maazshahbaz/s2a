import asyncio
import time

from pipeline import Pipeline


def callback(raw_trans, labeled_trans, analysis, metadata):
    """Process pipeline results."""
    # print("\n=== Labeled Transcription ===")
    print(labeled_trans)
    print(analysis)
    print("\n=== Metadata ===")
    # print(f"Chunks processed: {metadata['chunk_count']}")
    # print(f"Number of speakers: {metadata['num_speakers']}")


async def run_pipeline_async(audio_path: str, request_id: str):
    """Execute the pipeline asynchronously."""
    pipeline = Pipeline()
    return await pipeline.run_pipeline(audio_path, request_id)


if __name__ == "__main__":
    audio_path = "/home/sj/Desktop/data/back2/bytepulse-ai/uploads/2025-11-18/fe5cf860-62c3-45a0-8782-35fe0a482beb.wav"
    # audio_path = "/home/sj/Desktop/data/back2/bytepulse-ai/uploads/2025-11-18/2a0c5881-6982-469e-a59b-1a9d8469870f.wav"
    # audio_path = "/home/sj/Desktop/data/back2/bytepulse-ai/uploads/2025-11-18/0b5fc824-aaeb-4d9a-917d-6d3ae75c5efb.wav"
    # audio_path = "/home/sj/Desktop/data/back2/bytepulse-ai/data/s2a/corpus/customer_support/customer_support_data/out-9092701399-5079-20250721-214258-1753152178.1229.wav"
    # audio_path = "/home/sj/Desktop/data/back2/bytepulse-ai/s2a-omar-development/asr/clipped_60min.wav"
    # audio_path = "/home/sj/Desktop/data/back2/bytepulse-ai/s2a-omar-development/pipeline_dev/random_samples/in-9524528884-6154155856-20250320-165204-1742507524.88070.wav"
    # audio_path = "/home/sj/Desktop/data/back2/bytepulse-ai/s2a-omar-development/pipeline_dev/sales_records/out-9547070098-5075-20260101-113923-1767289163.118332.wav"
    # Run pipeline
    request_id = "1"
    start_time = time.time()
    
    raw_trans, labeled_trans, analysis, metadata = asyncio.run(
        run_pipeline_async(audio_path, request_id)
    )
    
    elapsed_time = time.time() - start_time
    
    # Process results
    callback(raw_trans, labeled_trans, analysis, metadata)
    
    print(f"\nTotal Time: {elapsed_time:.2f} seconds")
