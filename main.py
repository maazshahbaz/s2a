from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import os
from loguru import logger
import time
from contextlib import asynccontextmanager
from config import get_settings
from api.routers import all_routers
from generated.prisma import Prisma
from db_services.auth import initialize_auth_store
from intelligent_pipeline.pipeline import Pipeline

prisma = Prisma()

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup
    settings = get_settings()
    app.state.app_start_time = time.time()

    if not settings.staging_mode:
        logger.info("Connecting to database...")
        await prisma.connect()
        app.state.db = prisma
        logger.info("Database connected ")

        # Initialize auth store with database connection
        logger.info("Initializing authentication service...")
        initialize_auth_store(prisma)
        logger.info("Authentication service initialized ")
    else:
        logger.info("STAGING MODE — skipping database and auth")
        app.state.db = None
    if not settings.staging_mode:
        logger.info("Connecting to database...")
        await prisma.connect()
        app.state.db = prisma
        logger.info("Database connected ")

        # Initialize auth store with database connection
        logger.info("Initializing authentication service...")
        initialize_auth_store(prisma)
        logger.info("Authentication service initialized ")
    else:
        logger.info("STAGING MODE — skipping database and auth")
        app.state.db = None

    logger.info("Initializing Triton Inference Service...")

    try:
        app.state.triton_service = Pipeline()
    except Exception as e:
        logger.error(f"Failed to initialize TritonService: {e}")
        app.state.triton_service = None

    # Initialize streaming infrastructure
    logger.info("Initializing streaming services...")
    app.state.session_manager = None
    app.state.streaming_asr_client = None
    app.state.streaming_diar_client = None
    app.state.audiosocket_server = None

    try:
        from intelligent_pipeline.session_manager import SessionManager
        app.state.session_manager = SessionManager(
            max_concurrent=settings.streaming_max_concurrent_sessions,
            chunk_duration=settings.streaming_chunk_duration,
        )
    except Exception as e:
        logger.warning(f"Session manager unavailable: {e}")

    if app.state.session_manager:
        from intelligent_pipeline.streaming_asr_client import AsyncStreamingASRClient
        app.state.streaming_asr_client = AsyncStreamingASRClient()
        try:
            await app.state.streaming_asr_client.connect()
            logger.info("Streaming ASR client connected")
        except Exception as e:
            logger.warning(f"Streaming ASR not available: {e}")
            app.state.streaming_asr_client = None

        from intelligent_pipeline.streaming_diar_client import AsyncStreamingDiarClient
        app.state.streaming_diar_client = AsyncStreamingDiarClient()
        try:
            await app.state.streaming_diar_client.connect()
            logger.info("Streaming diarization client connected")
        except Exception as e:
            logger.warning(f"Streaming diarization not available: {e}")
            app.state.streaming_diar_client = None

    audiosocket_allowed = True
    if not settings.staging_mode and not settings.streaming_allowed_ips:
        audiosocket_allowed = False
        logger.warning(
            "AudioSocket server disabled in non-staging mode because STREAMING_ALLOWED_IPS is not configured"
        )

    if (
        app.state.session_manager
        and app.state.streaming_asr_client
        and app.state.streaming_diar_client
        and audiosocket_allowed
    ):
        try:
            from api.audiosocket_server import AudioSocketServer
            app.state.audiosocket_server = AudioSocketServer(
                session_manager=app.state.session_manager,
                streaming_asr=app.state.streaming_asr_client,
                streaming_diar=app.state.streaming_diar_client,
                port=settings.streaming_audiosocket_port,
                default_callback_url=settings.streaming_default_callback_url,
                staging_mode=settings.staging_mode,
                db=app.state.db,
                idle_timeout_seconds=settings.streaming_idle_timeout_seconds,
                max_session_duration_seconds=settings.streaming_max_session_duration_seconds,
                max_frame_bytes=settings.streaming_max_frame_bytes,
                max_bytes_per_second=settings.streaming_max_bytes_per_second,
                inference_timeout_seconds=settings.streaming_inference_timeout_seconds,
                allowed_ips=settings.streaming_allowed_ips,
            )
            await app.state.audiosocket_server.start()
            logger.info("AudioSocket server started")
        except Exception as e:
            logger.warning(f"AudioSocket server unavailable: {e}")
            app.state.audiosocket_server = None
    else:
        logger.warning("AudioSocket server not started: session manager, ASR, or diarization backend unavailable")

    if app.state.session_manager and app.state.streaming_asr_client and app.state.streaming_diar_client:
        logger.info("WebSocket streaming initialized")
    else:
        logger.warning("WebSocket streaming unavailable: session manager, ASR, or diarization backend not ready")

    yield
    
    # Shutdown
        
    # Shutdown streaming services
    if getattr(app.state, "audiosocket_server", None):
        await app.state.audiosocket_server.stop()
        logger.info("AudioSocket server stopped")
    if getattr(app.state, "streaming_asr_client", None):
        await app.state.streaming_asr_client.close()
        logger.info("Streaming ASR client closed")
    if getattr(app.state, "streaming_diar_client", None):
        await app.state.streaming_diar_client.close()
        logger.info("Streaming diarization client closed")

    if hasattr(app.state, "triton_service"):
        app.state.triton_service = None
        logger.info("Triton service released")
    
    if not settings.staging_mode:
        logger.info("Disconnecting database...")
        await prisma.disconnect()
        logger.info("Database disconnected")
    if not settings.staging_mode:
        logger.info("Disconnecting database...")
        await prisma.disconnect()
        logger.info("Database disconnected")
    
    logger.info("Microservice shutdown complete")

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
        "https://dev.api.bytepulseai.com", # Next.js production container
        "https://dev.bytepulseai.com", # Production domain
        "https://bytepulseai.com", # Production domain
        "https://api.bytepulseai.com", # Production domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Authorization", 
        "Content-Type",
        "x-user-id",
        "x-timestamp",
        "x-body-hash",
        "x-signature"
    ],
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
