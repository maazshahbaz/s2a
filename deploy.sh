#!/bin/bash

# S2A Speech-to-Actions Microservice Deployment Script
set -e

echo "🚀 Starting S2A Microservice Deployment"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi
print_status "Docker is installed"

# Check Docker Compose
if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not available"
    exit 1
fi
print_status "Docker Compose is available"

# Check NVIDIA Docker support
if ! docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    print_error "NVIDIA GPU support not available in Docker"
    print_warning "Make sure NVIDIA Container Toolkit is installed"
    exit 1
fi
print_status "NVIDIA GPU support is available"

# Check if services are already running
if docker compose ps -q | grep -q .; then
    print_warning "Services are already running. Stopping them first..."
    docker compose down
fi

# Build and start services
echo "🏗️  Building and starting services..."
docker compose up -d --build

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🩺 Checking service health..."

# Check main API service
if curl -f http://localhost:8001/health &> /dev/null; then
    print_status "S2A API service is healthy"
else
    print_error "S2A API service is not responding"
    echo "📋 Checking logs..."
    docker compose logs s2a-api
    exit 1
fi

# Check Prometheus (optional)
if curl -f http://localhost:9091 &> /dev/null; then
    print_status "Prometheus is running"
else
    print_warning "Prometheus is not responding (optional service)"
fi

# Check Grafana (optional) 
if curl -f http://localhost:3000 &> /dev/null; then
    print_status "Grafana is running"
else
    print_warning "Grafana is not responding (optional service)"
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📡 Service Endpoints:"
echo "   • S2A API:        http://localhost:8001"
echo "   • Health Check:   http://localhost:8001/health"
echo "   • API Stats:      http://localhost:8001/v1/stats"
echo "   • Prometheus:     http://localhost:9091"
echo "   • Grafana:        http://localhost:3000 (admin/admin)"
echo ""
echo "🧪 Test the API:"
echo '   curl -X POST "http://localhost:8001/v1/transcribe" \'
echo '     -H "Content-Type: multipart/form-data" \'
echo '     -F "audio_file=@your_audio.wav"'
echo ""
echo "📚 Documentation: API_USAGE.md"
echo "🔍 View logs: docker compose logs -f s2a-api"
echo "🛑 Stop services: docker compose down"