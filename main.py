from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import os
from loguru import logger
import time
from contextlib import asynccontextmanager
from services.asr_service import NeMoASRService
from services.audio_utils import AudioProcessor
from services.batch_processor import BatchProcessor, BatchProcessorConfig
from config import get_settings, get_intelligence_settings, get_redis_settings
from api.routers import all_routers
from intelligence.intelligence_service import start_intelligence_service, stop_intelligence_service
from generated.prisma import Prisma
from db_services.auth import initialize_auth_store

prisma = Prisma()

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup
    settings = get_settings()
    app.state.app_start_time = time.time()

    logger.info("Connecting to database...")
    await prisma.connect()
    app.state.db = prisma
    logger.info("Database connected ✅")
    
    # Initialize auth store with database connection
    logger.info("Initializing authentication service...")
    initialize_auth_store(prisma)
    logger.info("Authentication service initialized ✅")
    
    logger.info("Initializing ASR microservice...")
    
    # Initialize services
    app.state.asr_service = NeMoASRService(
        model_name=settings.model_name,
        device=settings.device,
        batch_size=settings.batch_size,
        max_chunk_duration=settings.max_chunk_duration,
        min_audio_duration=settings.min_audio_duration,
        overlap_duration=settings.overlap_duration,
        target_sample_rate=settings.target_sample_rate,
        words_per_second=settings.words_per_second,
        overlap_similarity_threshold=settings.overlap_similarity_threshold
    )
    
    app.state.audio_processor = AudioProcessor(
        target_sr=settings.target_sample_rate,
        vad_aggressiveness=settings.vad_aggressiveness
    )
    
    # Initialize Redis-based batch processor
    redis_settings = get_redis_settings()
    batch_config = BatchProcessorConfig(
        redis_host=redis_settings.host,
        redis_port=redis_settings.port,
        redis_db=redis_settings.db,
        redis_password=redis_settings.password,
        batch_size=redis_settings.batch_size,
        num_workers=redis_settings.num_workers,
        max_chunk_duration=settings.max_chunk_duration,
        overlap_duration=settings.overlap_duration,
        audio_cache_size=redis_settings.audio_cache_size
    )

    app.state.batch_processor = BatchProcessor(
        asr_service=app.state.asr_service,
        config=batch_config
    )
    
    # Start batch processor
    await app.state.batch_processor.start()

    # Initialize intelligence service if enabled
    intelligence_settings = get_intelligence_settings()
    if intelligence_settings.enabled:
        try:
            await start_intelligence_service()
            logger.info("Intelligence service started successfully")
        except Exception as e:
            logger.warning(f"Failed to start intelligence service: {e}")
            logger.info("Continuing without intelligence service")

    logger.info("ASR microservice initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ASR microservice...")

    # Stop intelligence service
    try:
        await stop_intelligence_service()
        logger.info("Intelligence service stopped")
    except Exception as e:
        logger.warning(f"Error stopping intelligence service: {e}")

    if app.state.batch_processor:
        await app.state.batch_processor.stop()
    
    logger.info("Disconnecting database...")
    await prisma.disconnect()
    logger.info("Database disconnected ✅")
    
    logger.info("ASR microservice shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="S2A Speech-to-Text Microservice",
    description="High-performance ASR service using NVIDIA NeMo Parakeet model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development
        "http://localhost:8080",  # Vue development
        "https://your-domain.com"  # Production domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

for router in all_routers:
    app.include_router(router, prefix=os.getenv("API_VERSION","/v1"))

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "BytePulse AI S2A Speech-to-Text Microservice", 
        "status": "running",
        "version": "1.0.0",
        "authentication": "required"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1,  # Single worker due to GPU memory constraints
        log_level="info"
    )