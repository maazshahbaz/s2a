#!/bin/bash
set -e

echo "Starting vLLM service for S2A Intelligence Pipeline..."
echo "Model: $MODEL_NAME"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Max Model Length: $MAX_MODEL_LEN"

# Wait for GPU to be available
echo "Checking GPU availability..."
nvidia-smi

# Start metrics collection in background
python /app/metrics.py &

# Pre-download model if not cached
echo "Checking model cache..."
if [ ! -d "/models/$MODEL_NAME" ]; then
    echo "Model not found in cache, will download on first request..."
fi

# Start vLLM server with optimized settings for S2A
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --port "$PORT" \
    --host "$HOST" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --disable-log-requests \
    --served-model-name "s2a-intelligence" \
    --max-num-batched-tokens 16384 \
    --max-paddings 64 \
    --enable-chunked-prefill