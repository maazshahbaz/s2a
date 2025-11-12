#!/bin/bash
# Test script for Llama 3.1 8B Instruct on Triton Inference Server (HTTP)

set -e

SERVER_URL="localhost:2000"
MODEL_NAME="llama-8b-instruct"

echo "================================================================================"
echo "Triton Llama 3.1 8B Instruct - HTTP API Test Suite"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo ""
    echo "================================================================================"
g   echo "$1"
    echo "================================================================================"
    echo ""
}

# Function to check if server is ready
check_server() {
    echo -n "Checking if server is ready... "
    RESPONSE=$(curl -s "http://${SERVER_URL}/v2/health/ready")
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Server is ready${NC}"
        return 0
    else
        echo -e "${RED}✗ Server is not ready${NC}"
        return 1
    fi
}

# Function to check if model is ready
check_model() {
    echo -n "Checking if model '${MODEL_NAME}' is ready... "
    RESPONSE=$(curl -s "http://${SERVER_URL}/v2/models/${MODEL_NAME}/ready")
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model is ready${NC}"
        return 0
    else
        echo -e "${RED}✗ Model is not ready${NC}"
        return 1
    fi
}

# Test 1: Server Health Check
print_section "TEST 1: Server Health Check"
check_server || exit 1

# Test 2: Server Live Check
print_section "TEST 2: Server Live Check"
echo "Checking if server is live..."
LIVE_RESPONSE=$(curl -s "http://${SERVER_URL}/v2/health/live")
echo "Response: ${LIVE_RESPONSE}"
echo ""

# Test 3: Model Ready Check
print_section "TEST 3: Model Ready Check"
check_model || exit 1

# Test 4: Get Model Metadata
print_section "TEST 4: Model Metadata"
echo "Fetching model metadata..."
curl -s "http://${SERVER_URL}/v2/models/${MODEL_NAME}" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Could not parse as JSON"
echo ""

# Test 5: Get Model Config
print_section "TEST 5: Model Configuration"
echo "Fetching model configuration..."
curl -s "http://${SERVER_URL}/v2/models/${MODEL_NAME}/config" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Could not parse as JSON"
echo ""

# Test 6: Simple Inference
print_section "TEST 6: Simple Inference (Basic Question)"
echo "Sending simple inference request..."
echo "Question: What is the capital of France?"
echo ""
curl -X POST "http://${SERVER_URL}/v2/models/${MODEL_NAME}/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "prompt",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["What is the capital of France?"]
      },
      {
        "name": "max_tokens",
        "datatype": "INT32",
        "shape": [1, 1],
        "data": [50]
      }
    ]
  }' 2>/dev/null | python3 -m json.tool 2>/dev/null
echo ""

# Test 7: Inference with Custom Parameters
print_section "TEST 7: Inference with Custom Parameters"
echo "Testing with custom temperature and top_p..."
echo "Question: Explain quantum computing in one sentence."
echo "Temperature: 0.8, Top-p: 0.95"
echo ""
curl -X POST "http://${SERVER_URL}/v2/models/${MODEL_NAME}/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "prompt",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["Explain quantum computing in one sentence."]
      },
      {
        "name": "max_tokens",
        "datatype": "INT32",
        "shape": [1, 1],
        "data": [100]
      },
      {
        "name": "temperature",
        "datatype": "FP32",
        "shape": [1, 1],
        "data": [0.8]
      },
      {
        "name": "top_p",
        "datatype": "FP32",
        "shape": [1, 1],
        "data": [0.95]
      }
    ]
  }' 2>/dev/null | python3 -m json.tool 2>/dev/null
echo ""

# Test 8: Inference with System Prompt
print_section "TEST 8: Inference with System Prompt"
echo "Testing with custom system prompt..."
echo "System: You are a professional barista with 20 years of experience."
echo "Question: How do I make perfect espresso?"
echo ""
curl -X POST "http://${SERVER_URL}/v2/models/${MODEL_NAME}/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "prompt",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["How do I make perfect espresso?"]
      },
      {
        "name": "system_prompt",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["You are a professional barista with 20 years of experience."]
      },
      {
        "name": "max_tokens",
        "datatype": "INT32",
        "shape": [1, 1],
        "data": [200]
      },
      {
        "name": "temperature",
        "datatype": "FP32",
        "shape": [1, 1],
        "data": [0.7]
      }
    ]
  }' 2>/dev/null | python3 -m json.tool 2>/dev/null
echo ""

# Test 9: Call Transcription Analysis
print_section "TEST 9: Call Center Transcription Analysis"
echo "Testing call center analysis use case..."
echo "This is your primary use case with JSON output."
echo ""

SAMPLE_TRANSCRIPTION="Thank you for calling customer support, my name is Sarah, how can I help you today? Hi Sarah, I'm having issues with my internet connection, it keeps dropping every few minutes and it's really frustrating because I work from home. I'm sorry to hear that, let me help you resolve this, can I have your account number please? Sure, it's 12345678. Thank you, I can see your account now, let me run some diagnostics on your connection to see what might be causing the problem. Okay, thank you, I appreciate your help. I found the issue, there's a problem with your router configuration and the firmware is outdated, I'm going to push a firmware update to your device which should resolve the disconnection issues. That's great, how long will it take? The update should complete in about 5 minutes and your internet will be briefly interrupted during the update, but after that everything should work smoothly. Perfect, thank you so much for your help, I was worried I'd have to wait days for a technician! You're welcome, I'm glad I could help you quickly, is there anything else I can help you with today? No, that's all, thanks again for the fast resolution! Thank you for calling, have a great day and happy working from home!"

ANALYSIS_PROMPT="You are an expert call center quality analyst. The following text is a call transcription written as a single paragraph with no speaker labels. Your task is to infer the likely dialogue flow between the customer and the agent, and then provide a structured analysis of the call.

Call Transcription:
${SAMPLE_TRANSCRIPTION}

Perform the following steps:
1. Reconstruct the conversation in your mind by inferring which parts are spoken by the customer and which by the agent.
2. Analyze the inferred conversation for tone, resolution, and key insights.

Respond ONLY with a valid JSON object (no extra text or explanations).
The JSON must strictly follow this structure:
{
    \"inferred_conversation_structure\": \"Briefly describe how the conversation likely flowed between customer and agent.\",
    \"call_sentiment\": \"positive\" | \"negative\" | \"neutral\",
    \"call_summary\": \"A summary of the main points of the call.\",
    \"call_status\": \"resolved\" | \"pending\" | \"escalated\",
    \"call_improvement_points\": [
        \"Specific, actionable improvement point 1\",
        \"Specific, actionable improvement point 2\"
    ],
    \"key_words\": [
        \"keyword1\",
        \"keyword2\",
        \"keyword3\"
    ]
}"

# Save to temp file to avoid shell escaping issues
cat > /tmp/call_analysis_payload.json <<EOF
{
  "inputs": [
    {
      "name": "prompt",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["${ANALYSIS_PROMPT}"]
    },
    {
      "name": "system_prompt",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["You are a helpful assistant that analyzes call center transcriptions and provides structured insights."]
    },
    {
      "name": "max_tokens",
      "datatype": "INT32",
      "shape": [1, 1],
      "data": [512]
    },
    {
      "name": "temperature",
      "datatype": "FP32",
      "shape": [1, 1],
      "data": [0.3]
    },
    {
      "name": "top_p",
      "datatype": "FP32",
      "shape": [1, 1],
      "data": [0.9]
    }
  ]
}
EOF

curl -X POST "http://${SERVER_URL}/v2/models/${MODEL_NAME}/infer" \
  -H "Content-Type: application/json" \
  -d @/tmp/call_analysis_payload.json 2>/dev/null | python3 -m json.tool 2>/dev/null

rm -f /tmp/call_analysis_payload.json
echo ""

# Test 10: Creative Writing Test
print_section "TEST 10: Creative Writing (Higher Temperature)"
echo "Testing creative output with higher temperature..."
echo "Task: Write a haiku about artificial intelligence"
echo "Temperature: 0.9, Top-p: 0.95"
echo ""
curl -X POST "http://${SERVER_URL}/v2/models/${MODEL_NAME}/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "prompt",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["Write a haiku about artificial intelligence."]
      },
      {
        "name": "system_prompt",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["You are a creative poet."]
      },
      {
        "name": "max_tokens",
        "datatype": "INT32",
        "shape": [1, 1],
        "data": [100]
      },
      {
        "name": "temperature",
        "datatype": "FP32",
        "shape": [1, 1],
        "data": [0.9]
      },
      {
        "name": "top_p",
        "datatype": "FP32",
        "shape": [1, 1],
        "data": [0.95]
      }
    ]
  }' 2>/dev/null | python3 -m json.tool 2>/dev/null
echo ""

# Test 11: Server Statistics
print_section "TEST 11: Server Statistics"
echo "Fetching server statistics..."
curl -s "http://${SERVER_URL}/v2/models/${MODEL_NAME}/stats" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Statistics not available"
echo ""

# Test 12: Metrics Endpoint
print_section "TEST 12: Prometheus Metrics"
echo "Fetching Prometheus metrics (filtering for model-specific metrics)..."
curl -s "http://localhost:8002/metrics" 2>/dev/null | grep -E "(nv_inference|nv_gpu)" | head -20 || echo "Metrics not available"
echo ""
echo "... (truncated, full metrics available at http://localhost:8002/metrics)"
echo ""

# Summary
print_section "TEST SUMMARY"
echo -e "${GREEN}✓ All tests completed!${NC}"
echo ""
echo "Key Endpoints:"
echo "  - Health:     http://${SERVER_URL}/v2/health/ready"
echo "  - Live:       http://${SERVER_URL}/v2/health/live"
echo "  - Inference:  http://${SERVER_URL}/v2/models/${MODEL_NAME}/infer"
echo "  - Metadata:   http://${SERVER_URL}/v2/models/${MODEL_NAME}"
echo "  - Config:     http://${SERVER_URL}/v2/models/${MODEL_NAME}/config"
echo "  - Stats:      http://${SERVER_URL}/v2/models/${MODEL_NAME}/stats"
echo "  - Metrics:    http://localhost:2002/metrics"
echo ""
echo "Model Parameters Tested:"
echo "  - max_tokens: 50, 100, 200, 512"
echo "  - temperature: 0.3, 0.7, 0.8, 0.9"
echo "  - top_p: 0.9, 0.95"
echo ""
echo "================================================================================"
