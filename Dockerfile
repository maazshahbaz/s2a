FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NODE_VERSION=20.19.4

# Install system dependencies + Node.js (needed for Prisma)
RUN apt-get update && apt-get install -y \
  ffmpeg \
  libsndfile1 \
  libsox-dev \
  sox \
  git \
  curl \
  build-essential \
  && curl -fsSL https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz \
  | tar -xJ -C /usr/local --strip-components=1 \
  && rm -rf /var/lib/apt/lists/*

# Verify node version
RUN node -v && npm -v

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (matching CUDA 12.8), torchvision, and torchaudio
RUN pip install --no-cache-dir \
  torch==2.9.0+cu128 \
  torchvision==0.24.0+cu128 \
  torchaudio==2.9.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# Install Ninja (needed for CUDA extensions)
RUN pip install --no-cache-dir ninja

# Clone and install Apex (for amp_C)
RUN git clone https://github.com/NVIDIA/apex.git /tmp/apex \
  && cd /tmp/apex \
  && pip install --no-cache-dir --no-build-isolation . --global-option="--cpp_ext" --global-option="--cuda_ext" \
  && cd /app && rm -rf /tmp/apex

# Install NeMo ASR and Megatron-Core
RUN pip install --no-cache-dir "nemo_toolkit[asr]" "megatron-core"

# Install Prisma CLI globally
RUN npm install -g prisma

# Copy application code
COPY . .

# Generate Prisma client
RUN prisma generate

# Create directories for temporary files and logs
RUN mkdir -p /tmp/s2a /app/logs

# Set permissions
RUN chmod +x /app

# Expose ports
EXPOSE 8001 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/v1/statistics/health || exit 1

# Default command
CMD ["sh", "-c", "prisma migrate deploy && prisma py fetch && python -m uvicorn main:app --host 0.0.0.0 --port 8001"]
