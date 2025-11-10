
import requests
import json
import time


class CallAnalysisClient:
    def __init__(self, server_url="http://localhost:2000", model_name="llama-8b-instruct"):
        """
        Initialize client with server URL and model name
        
        Args:
            server_url: Base URL of Triton server (default: http://localhost:2000)
            model_name: Name of the model (default: llama-8b-instruct)
        """
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.infer_url = f"{self.server_url}/v2/models/{self.model_name}/infer"
        self.metadata_url = f"{self.server_url}/v2/models/{self.model_name}"
        self.health_url = f"{self.server_url}/v2/health/ready"

    def check_health(self):
        """Check if server is healthy"""
        try:
            response = requests.get(self.health_url, timeout=5)
            print(self.health_url)
            if response.status_code == 200:
                print("Server is healthy")
                return True
            else:
                print(f"Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False

    def analyze_call(self, transcription, max_tokens=512, temperature=0.3, 
                     top_p=0.9, top_k=50):
        """
        Analyze call transcription and return structured JSON
        
        Args:
            transcription: Call transcription text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            verbose: Print detailed output
            
        Returns:
            dict: Parsed analysis result
        """

        # Construct the analysis prompt
        prompt = f"""                                                                                                                        You are an expert call center quality analyst. The following text is a call transcription written as a single paragraph with no speaker labels.
    Your task is to infer the likely dialogue flow between the customer and the agent, and then provide a structured analysis of the call.

    Call Transcription:
    {transcription}

    Perform the following steps:
    1. Reconstruct the conversation in your mind by inferring which parts are spoken by the customer and which by the agent.
    2. Analyze the inferred conversation for tone, resolution, and key insights.

    Respond ONLY with a valid JSON object (no extra text or explanations).

    The JSON must strictly follow this structure:
    {{
        "call_sentiment": "positive" | "negative" | "neutral",
        "call_summary": "A summary of the main points of the call.",
        "call_status": "resolved" | "pending" | "escalated",
        "call_improvement_points": [
            "Specific, actionable improvement point 1",
            "Specific, actionable improvement point 2"
        ],
        "key_words": [
            "keyword1",
            "keyword2",
            "keyword3"
        ]
    }}
    """

        system_prompt = "You are a helpful assistant that analyzes call center transcriptions and provides structured insights."

        # Build the request payload
        payload = {
            "inputs": [
                {
                    "name": "prompt",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [prompt]
                },
                {
                    "name": "system_prompt",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [system_prompt]
                },
                {
                    "name": "max_tokens",
                    "datatype": "INT32",
                    "shape": [1, 1],
                    "data": [max_tokens]
                },
                {
                    "name": "temperature",
                    "datatype": "FP32",
                    "shape": [1, 1],
                    "data": [temperature]
                },
                {
                    "name": "top_p",
                    "datatype": "FP32",
                    "shape": [1, 1],
                    "data": [top_p]
                },
                {
                    "name": "top_k",
                    "datatype": "INT32",
                    "shape": [1, 1],
                    "data": [top_k]
                }
            ]
        }

        try:
            # Send POST request
            start_time = time.time()
            response = requests.post(
                self.infer_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minute timeout for inference
            )
            elapsed_time = time.time() - start_time

            # Check response status
            if response.status_code != 200:
                print(f"✗ Request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return {"error": f"HTTP {response.status_code}", "details": response.text}

            # Parse response
            response_data = response.json()

            # Extract generated text from response
            generated_text = None
            if "outputs" in response_data:
                for output in response_data["outputs"]:
                    if output["name"] == "generated_text":
                        generated_text = output["data"][0]
                        break

            if not generated_text:
                return {"error": "No generated_text in response", "raw_response": response_data}


            # Parse JSON from generated text
            try:
                # Extract JSON if there's extra text
                json_start = generated_text.find('{')
                json_end = generated_text.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = generated_text[json_start:json_end]
                    result = json.loads(json_str)

                    # Add metadata
                    result["_metadata"] = {
                        "inference_time_seconds": elapsed_time,

                    }

                    return result
                else:

                    return {"error": "No valid JSON in response", "raw_text": generated_text}

            except json.JSONDecodeError as e:
                print(e)
                return {"error": "Invalid JSON", "raw_text": generated_text}

        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

        except Exception as e:

            return {"error": str(e)}


def main():
    # Sample call transcription
    sample_transcription = """
    Thank you for calling customer support, my name is Sarah, how can I help you today? Hi Sarah, I'm having issues with my internet connection, it keeps dropping every few minutes and it's really frustrating because I work from home. I'm sorry to hear that, let me help you resolve this, can I have your account number please? Sure, it's 12345678. Thank you, I can see your account now, let me run some diagnostics on your connection to see what might be causing the problem. Okay, thank you, I appreciate your help. I found the issue, there's a problem with your router configuration and the firmware is outdated, I'm going to push a firmware update to your device which should resolve the disconnection issues. That's great, how long will it take? The update should complete in about 5 minutes and your internet will be briefly interrupted during the update, but after that everything should work smoothly. Perfect, thank you so much for your help, I was worried I'd have to wait days for a technician! You're welcome, I'm glad I could help you quickly, is there anything else I can help you with today? No, that's all, thanks again for the fast resolution! Thank you for calling, have a great day and happy working from home!
    """
    
    # Initialize client
    print("Connecting to Triton Inference Server...")
    client = CallAnalysisClient(
        server_url="http://localhost:2000",
        model_name="llama-8b-instruct"
    )
    
    # Check server health
    if not client.check_health():
        print("\nServer is not available. Please start the Triton server first.")
        return

    # Analyze the call
    result = client.analyze_call(
        transcription=sample_transcription,
        max_tokens=512,
        temperature=0.3,
        top_p=0.9,
        top_k=50
    )

    # Display formatted results
    if "error" not in result:
        print(result)

    else:
        print(f"\nError occurred: {result.get('error')}")


if __name__ == "__main__":
    main()

