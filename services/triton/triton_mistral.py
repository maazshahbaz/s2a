import tritonclient.grpc as grpcclient
import numpy as np
import json
import uuid
from typing import Optional


class TritonMistralClient:
    def __init__(
        self,
        triton_url: str = "localhost:2001",
        model_name: str = "mistral-nemo"
    ):
        """
        Initialize Triton gRPC client
        
        Args:
            triton_url: Triton server URL (gRPC port)
            model_name: Model name in Triton
        """
        self.model_name = model_name
        self.client = grpcclient.InferenceServerClient(url=triton_url)
        
        # Check if model is loaded
        if not self.client.is_model_ready(self.model_name):
            raise Exception(f"Model {self.model_name} is not ready")
        
        print(f"✅ Connected to Triton gRPC server. Model {self.model_name} is ready.")
    
    def _callback(self, result, error):
        """
        Callback function for async inference - prints result as JSON
        
        Args:
            result: InferResult object
            error: Error object if request failed
        """
        request_id = result.get_response().id
        print(request_id)
        
        if error:
            print(json.dumps({
                "request_id": request_id,
                "success": False,
                "error": str(error)
            }, indent=2))
        else:
            # Get output text
            output_text = result.as_numpy("generated_text")[0].decode('utf-8')
            print(output_text)
            
            # Decode if bytes
            if isinstance(output_text, bytes):
                output_text = output_text.decode('utf-8')
            
            # Try to parse as JSON for analysis results
            try:
                output_text_cleaned = output_text.strip()
                if output_text_cleaned.startswith('```json'):
                    output_text_cleaned = output_text_cleaned.replace('```json', '').replace('```', '').strip()
                elif output_text_cleaned.startswith('```'):
                    output_text_cleaned = output_text_cleaned.replace('```', '').strip()
                
                start_idx = output_text_cleaned.find('{')
                end_idx = output_text_cleaned.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = output_text_cleaned[start_idx:end_idx]
                    analysis = json.loads(json_str)
                    
                    print(json.dumps({
                        "request_id": request_id,
                        "success": True,
                        "analysis": analysis
                    }, indent=2))
                else:
                    # Not JSON, just print the text
                    print(json.dumps({
                        "request_id": request_id,
                        "success": True,
                        "generated_text": output_text
                    }, indent=2))
                    
            except json.JSONDecodeError:
                # Not valid JSON, just print the text
                print(json.dumps({
                    "request_id": request_id,
                    "success": True,
                    "generated_text": output_text
                }, indent=2))
    
    def generate_async(self, prompt: str, request_id: Optional[str] = None, on_complete=None) -> str:
        """
        Generate text from a single prompt using async inference with callback
        
        Args:
            prompt: Input text prompt
            request_id: Optional request ID (will be generated if not provided)
            
        Returns:
            request_id: The request ID
        """
        # Generate request_id if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        print(f"📝 Sending prompt to Triton [ID: {request_id}]:\n{prompt}\n" + "-"*50)
        
        # Prepare input
        input_data = np.array([[prompt]], dtype=object)
        inputs = [grpcclient.InferInput("prompt", [1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(input_data)
        
        # Request output
        outputs = [grpcclient.InferRequestedOutput("generated_text")]
        
        # Run async inference with callback
        self.client.async_infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            callback=on_complete,
            request_id=request_id
        )
        
        print(f"📤 Sent async request: {request_id}")
        return request_id
    
    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format prompts according to Mistral's instruction format"""
        return f"""[INST] {system_prompt}

{user_prompt} [/INST]"""
    
    def analyze_call_async(self, transcription: str, request_id: Optional[str] = None, on_complete=None) -> str:
        """
        Analyze call center transcription asynchronously
        
        Args:
            transcription: Call transcription text
            request_id: Optional request ID (will be generated if not provided)
        
        Returns:
            request_id
        """
        system_prompt = """You are an expert call center quality analyst with years of experience evaluating customer service interactions. You specialize in analyzing unstructured call transcriptions, identifying customer sentiment, evaluating agent performance, and providing actionable feedback. You always provide responses in valid JSON format without any additional text or explanations."""
        
        user_prompt = f"""Analyze the following call center transcription. The transcription is provided as continuous text without speaker labels.

<transcription>
{transcription}
</transcription>

INSTRUCTIONS:
1. Identify who is the customer and who is the agent based on context clues
2. Determine the overall sentiment of the call from the customer's perspective
3. Assess whether the issue was resolved, is pending, or needs escalation
4. Extract 3-5 specific keywords that represent the main topics discussed
5. Identify 2-4 concrete, actionable improvement points for the agent

RESPONSE FORMAT:
Return ONLY a valid JSON object with this exact structure:

{{
    "call_sentiment": "<positive|negative|neutral>",
    "call_summary": "<2-3 sentence summary>",
    "call_status": "<resolved|pending|escalated>",
    "call_improvement_points": [
        "<improvement 1>",
        "<improvement 2>"
    ],
    "key_words": [
        "<keyword1>",
        "<keyword2>",
        "<keyword3>"
    ]
}}

Output ONLY the JSON object, nothing else."""
        
        full_prompt = self.format_prompt(system_prompt, user_prompt)
        
        # Generate response asynchronously
        return self.generate_async(full_prompt, request_id, on_complete)
    
    def close(self):
        """Close the client connection"""
        self.client.close()


# Usage Example
if __name__ == "__main__":
    import time
    
    # Initialize client
    client = TritonMistralClient(
        triton_url="localhost:2001",
        model_name="mistral-nemo"
    )
    
    # Test transcription
    transcription = """
    Hello thank you for calling TechSupport how can I help you today yes hi I'm having trouble 
    with my internet connection it keeps dropping every few minutes I'm so sorry to hear that 
    let me help you with that can you tell me what type of router you have it's a NetGear 
    Nighthawk okay great and when did this issue start it started about two days ago alright 
    I'd like to try a few troubleshooting steps first can you please unplug your router for 
    30 seconds okay I've unplugged it should I plug it back in now yes please plug it back in 
    and let's wait for all the lights to come back on okay all the lights are back on now 
    great can you try connecting to the internet now oh wow it's working perfectly now that's 
    wonderful I'm glad we could resolve this for you is there anything else I can help you 
    with today no that's all thank you so much for your help you're very welcome have a great 
    day you too bye
    """
    
    print("\n" + "="*80)
    print("Single async call analysis")
    print("="*80)
    
    # Send async request - result will be printed by callback
    request_id = client.analyze_call_async(transcription)
    
    # Wait for callback to complete
    time.sleep(6)
    
    print("\n" + "="*80)
    print("Batch async processing")
    print("="*80)
    
    # Send multiple requests
    transcriptions = [transcription, transcription, transcription]
    
    for i, trans in enumerate(transcriptions):
        client.analyze_call_async(trans)
        print(f"Sent batch request {i+1}/{len(transcriptions)}")
    
    # Wait for all callbacks to complete
    print("\nWaiting for results (callbacks will print automatically)...")
    time.sleep(6)
    
    # Close connection
    client.close()
    print("\n✅ Client closed")