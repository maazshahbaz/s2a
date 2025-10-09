#!/usr/bin/env python3
"""
Health check script for vLLM service
"""

import sys
import httpx
import time

def check_health():
    """Check if vLLM service is healthy"""
    try:
        # Check if the API is responding
        with httpx.Client(timeout=10.0) as client:
            # Check models endpoint
            models_response = client.get("http://localhost:8000/v1/models")
            if models_response.status_code != 200:
                print(f"Models endpoint failed: {models_response.status_code}")
                return False

            # Quick completion test
            test_payload = {
                "model": "s2a-intelligence",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "temperature": 0.1
            }

            completion_response = client.post(
                "http://localhost:8000/v1/chat/completions",
                json=test_payload,
                timeout=30.0
            )

            if completion_response.status_code != 200:
                print(f"Completion endpoint failed: {completion_response.status_code}")
                return False

            print("vLLM service is healthy")
            return True

    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)