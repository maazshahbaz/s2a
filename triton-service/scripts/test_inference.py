#!/usr/bin/env python3
from client_triton import TritonClient

client = TritonClient(
        url="localhost:2001",  # gRPC port
        model_name="llama-8b-instruct"
    )
def main():
    # Sample transcription
    sample_transcription = """
    Thank you for calling customer support, my name is Sarah, how can I help you today? Hi Sarah, I'm having issues with my internet connection, it keeps dropping every few minutes and it's really frustrating because I work from home. I'm sorry to hear that, let me help you resolve this, can I have your account number please? Sure, it's 12345678. Thank you, I can see your account now, let me run some diagnostics on your connection to see what might be causing the problem. Okay, thank you, I appreciate your help. I found the issue, there's a problem with your router configuration and the firmware is outdated, I'm going to push a firmware update to your device which should resolve the disconnection issues. That's great, how long will it take? The update should complete in about 5 minutes and your internet will be briefly interrupted during the update, but after that everything should work smoothly. Perfect, thank you so much for your help, I was worried I'd have to wait days for a technician! You're welcome, I'm glad I could help you quickly, is there anything else I can help you with today? No, that's all, thanks again for the fast resolution! Thank you for calling, have a great day and happy working from home!
    """

    # Initialize client
    print("Connecting to Triton Inference Server via gRPC...")


    # Check server health
    try:
        if client.client.is_server_live():
            print("Triton server is live")
        if client.client.is_server_ready():
            print("Triton server is ready")
        if client.client.is_model_ready(client.model_name):
            print(f"Model '{client.model_name}' is ready\n")
        else:
            print(f"Model '{client.model_name}' is not ready")
            return
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # Analyze the call
    result = client.analyze_call(
        transcription=sample_transcription,
        max_tokens=512,
        temperature=0.3,  # Lower temperature for more consistent JSON
        top_p=0.9
    )

    # Display results in a formatted way
    if "error" not in result:
        print(result)
    else:
        print(f"\n Error occurred: {result.get('error')}")


if __name__ == "__main__":
    main()

