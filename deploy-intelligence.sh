#!/bin/bash
set -e

echo "🚀 Deploying S2A with Enhanced Intelligence Pipeline"

# Check if Docker and docker-compose are available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed"
    exit 1
fi

# Check if NVIDIA Docker is available
if ! docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker runtime is not available"
    echo "Please install nvidia-container-toolkit"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs vllm-logs monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources

# Set proper permissions
chmod +x vllm-config/startup.sh

# Check available disk space
AVAILABLE_SPACE=$(df /mnt/storage | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_SPACE" -lt 50000000 ]; then  # 50GB
    echo "⚠️  Warning: Less than 50GB available space on /mnt/storage"
    echo "   Model downloads may require significant space"
fi

# Check GPU memory
echo "🔍 Checking GPU resources..."
nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating default .env file..."
    cat > .env << EOF
# API Key Secret (CHANGE THIS!)
API_KEY_SECRET=change-me-to-secure-32-char-secret

# S2A Configuration
S2A_MODEL_NAME=nvidia/parakeet-tdt-0.6b-v2
S2A_DEVICE=cuda
S2A_BATCH_SIZE=8
S2A_GPU_MEMORY_FRACTION=0.4
S2A_NUM_WORKERS=2

# Intelligence Configuration
S2A_INTEL_ENABLED=true
S2A_INTEL_VLLM_BASE_URL=http://vllm-service:8000/v1
S2A_INTEL_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
S2A_INTEL_AUTO_PROCESS=true

# Database (if using PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost:5432/s2a
EOF
    echo "⚠️  Please edit .env file with your configuration"
fi

# Deployment options
echo "🔧 Deployment Options:"
echo "1. Full stack (S2A + vLLM + Monitoring)"
echo "2. S2A + vLLM only"
echo "3. S2A only (existing deployment)"
echo "4. vLLM only"

read -p "Choose deployment option (1-4): " OPTION

case $OPTION in
    1)
        echo "🚀 Deploying full stack..."
        docker-compose -f docker-compose.intelligence.yml up -d
        ;;
    2)
        echo "🚀 Deploying S2A + vLLM..."
        docker-compose -f docker-compose.intelligence.yml up -d s2a-api vllm-service redis-cache
        ;;
    3)
        echo "🚀 Deploying S2A only..."
        docker-compose -f docker-compose.yml up -d
        ;;
    4)
        echo "🚀 Deploying vLLM only..."
        docker-compose -f docker-compose.intelligence.yml up -d vllm-service
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo "⏳ Waiting for services to start..."

# Wait for services to be healthy
if [ "$OPTION" != "4" ]; then
    echo "⏳ Waiting for S2A service..."
    while ! curl -f http://localhost:8001/v1/statistics/health &> /dev/null; do
        echo "   S2A service not ready, waiting..."
        sleep 10
    done
    echo "✅ S2A service is healthy"
fi

if [ "$OPTION" != "3" ]; then
    echo "⏳ Waiting for vLLM service..."
    while ! curl -f http://localhost:8000/v1/models &> /dev/null; do
        echo "   vLLM service not ready, waiting..."
        sleep 15
    done
    echo "✅ vLLM service is healthy"
fi

echo ""
echo "🎉 Deployment Complete!"
echo ""
echo "📋 Service URLs:"
if [ "$OPTION" != "4" ]; then
    echo "   S2A API: http://localhost:8001"
    echo "   S2A Health: http://localhost:8001/v1/statistics/health"
    echo "   S2A Metrics: http://localhost:9090/metrics"
fi

if [ "$OPTION" != "3" ]; then
    echo "   vLLM API: http://localhost:8000"
    echo "   vLLM Models: http://localhost:8000/v1/models"
fi

if [ "$OPTION" = "1" ]; then
    echo "   Prometheus: http://localhost:9091"
    echo "   Grafana: http://localhost:3000 (admin/admin)"
fi

echo ""
echo "🔧 Management Commands:"
echo "   View logs: docker-compose -f docker-compose.intelligence.yml logs -f"
echo "   Stop services: docker-compose -f docker-compose.intelligence.yml down"
echo "   Restart: docker-compose -f docker-compose.intelligence.yml restart"
echo ""
echo "💡 Next Steps:"
echo "   1. Create an API key: python key_manager.py create --name test-key --type project"
echo "   2. Test transcription: curl -X POST http://localhost:8001/v1/transcription/transcribe -H 'Authorization: Bearer YOUR_KEY' -F 'audio_file=@test.wav'"
echo "   3. Test intelligence: curl -X POST http://localhost:8001/v1/intelligence/extract/sync -H 'Authorization: Bearer YOUR_KEY' -d '{\"transcript_id\":\"test\",\"transcript_text\":\"Hello world\"}'"