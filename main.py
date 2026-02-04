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

    logger.info("Connecting to database...")
    await prisma.connect()
    app.state.db = prisma
    logger.info("Database connected ✅")
    
    # Initialize auth store with database connection
    logger.info("Initializing authentication service...")
    initialize_auth_store(prisma)
    logger.info("Authentication service initialized ✅")

    logger.info("Initializing Triton Inference Service...")

    try:
        app.state.triton_service = Pipeline()
    except Exception as e:
        logger.error(f"Failed to initialize TritonService: {e}")
        app.state.triton_service = None
    
    
    yield
    
    # Shutdown
        
    if hasattr(app.state, "triton_service"):
        app.state.triton_service = None
        logger.info("Triton service released ✅")
    
    logger.info("Disconnecting database...")
    await prisma.disconnect()
    logger.info("Database disconnected ✅")
    
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