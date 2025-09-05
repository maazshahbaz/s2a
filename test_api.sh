#!/bin/bash

# Create API key and store it
API_KEY=$(python3 -c "
from auth import api_key_store, APIKeyType
import sys
try:
    api_key, key_info = api_key_store.create_key('test-key', APIKeyType.PROJECT)
    print(api_key)
except Exception as e:
    print('Error creating key:', e, file=sys.stderr)
    sys.exit(1)
")

if [ $? -ne 0 ]; then
    echo "Failed to create API key"
    exit 1
fi

echo "Created API Key: $API_KEY"

# Create a dummy WAV file for testing
echo "Creating test audio file..."
ffmpeg -f lavfi -i "sine=frequency=440:duration=6" -ar 16000 -ac 1 test_audio.wav -y 2>/dev/null

if [ ! -f test_audio.wav ]; then
    echo "Warning: Could not create test audio file. Using /dev/null"
    TEST_FILE="/dev/null"
else
    TEST_FILE="test_audio.wav"
    echo "Created test_audio.wav"
fi

# Test the API
echo "Testing transcription API..."
curl -X POST "http://localhost:8001/v1/transcribe" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@$TEST_FILE" \
  -F "enhance_audio=true" \
  -F "remove_silence=false" \
  -v

# Clean up
if [ -f test_audio.wav ]; then
    rm test_audio.wav
fi