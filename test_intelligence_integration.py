#!/usr/bin/env python3
"""
Test script for integrated transcription + intelligence pipeline
Tests both sync and async endpoints with intelligence processing
"""

import asyncio
import httpx
import json
import time
from pathlib import Path
import sys

# Configuration
API_BASE_URL = "http://localhost:8001/v1"
API_KEY = "bp-proj-test"  # Replace with your API key
TEST_AUDIO_FILE = "test_audio.wav"  # Replace with your test audio file
WEBHOOK_URL = "https://webhook.site/your-unique-url"  # Replace with your webhook URL


async def test_sync_transcription_with_intelligence():
    """Test synchronous transcription with quick intelligence"""
    print("\n" + "="*60)
    print("Testing SYNC Transcription with Intelligence")
    print("="*60)

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Prepare the request
        with open(TEST_AUDIO_FILE, "rb") as f:
            files = {"audio_file": (TEST_AUDIO_FILE, f, "audio/wav")}
            data = {
                "enhance_audio": "true",
                "include_intelligence": "true",
                "intelligence_mode": "auto_detect"
            }
            headers = {"Authorization": f"Bearer {API_KEY}"}

            # Send request
            print(f"Sending audio file: {TEST_AUDIO_FILE}")
            start_time = time.time()

            response = await client.post(
                f"{API_BASE_URL}/transcription/transcribe",
                files=files,
                data=data,
                headers=headers
            )

            elapsed = time.time() - start_time
            print(f"Response received in {elapsed:.2f}s")

            if response.status_code == 200:
                result = response.json()
                print(f"\nStatus: {result['status']}")
                print(f"Job ID: {result['job_id']}")
                print(f"Duration: {result.get('duration', 0):.2f}s")
                print(f"RTF: {result.get('rtf', 0):.3f}")

                # Display transcript
                if result.get('text'):
                    print(f"\nTranscript (first 200 chars):")
                    print(f"  {result['text'][:200]}...")

                # Display quick intelligence if available
                if result.get('quick_intelligence'):
                    qi = result['quick_intelligence']
                    print(f"\n✨ Quick Intelligence (received in {qi.get('processing_time', 0):.2f}s):")
                    print(f"  Summary: {qi['summary']}")
                    print(f"  Intent: {qi['intent']}")
                    print(f"  Sentiment: {qi['sentiment']}")

                    if qi.get('action_items'):
                        print(f"  Action Items ({len(qi['action_items'])}):")
                        for item in qi['action_items'][:3]:
                            print(f"    - {item.get('task', 'N/A')}")

                    if qi.get('key_entities'):
                        print(f"  Key Entities: {', '.join(qi['key_entities'][:5])}")

                # Display enhanced intelligence status
                if result.get('enhanced_intelligence_status'):
                    eis = result['enhanced_intelligence_status']
                    print(f"\n🔄 Enhanced Intelligence Status:")
                    print(f"  Job ID: {eis['job_id']}")
                    print(f"  Status: {eis['status']}")
                    print(f"  Est. Completion: {eis.get('estimated_completion', 'N/A')}")

            else:
                print(f"❌ Error {response.status_code}: {response.text}")

            return result if response.status_code == 200 else None


async def test_async_transcription_with_intelligence():
    """Test asynchronous transcription with progressive intelligence webhooks"""
    print("\n" + "="*60)
    print("Testing ASYNC Transcription with Intelligence")
    print("="*60)

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Prepare the request
        with open(TEST_AUDIO_FILE, "rb") as f:
            files = {"audio_file": (TEST_AUDIO_FILE, f, "audio/wav")}
            data = {
                "callback_url": WEBHOOK_URL,
                "enhance_audio": "true",
                "include_intelligence": "true",
                "intelligence_mode": "auto_detect",
                "priority": "0"
            }
            headers = {"Authorization": f"Bearer {API_KEY}"}

            # Send request
            print(f"Sending audio file: {TEST_AUDIO_FILE}")
            print(f"Webhook URL: {WEBHOOK_URL}")

            response = await client.post(
                f"{API_BASE_URL}/transcription/transcribe/async",
                files=files,
                data=data,
                headers=headers
            )

            if response.status_code == 200:
                result = response.json()
                job_id = result['job_id']
                print(f"✅ Job submitted: {job_id}")
                print(f"Status: {result['status']}")

                # Monitor job status
                print("\n🔄 Monitoring job status...")
                for i in range(30):  # Check for 30 seconds
                    await asyncio.sleep(2)

                    # Check job status
                    status_response = await client.get(
                        f"{API_BASE_URL}/transcription/status/{job_id}",
                        headers=headers
                    )

                    if status_response.status_code == 200:
                        status = status_response.json()
                        print(f"  [{i*2}s] Status: {status['status']}")

                        if status['status'] == 'completed':
                            print("\n✅ Transcription completed!")
                            if status.get('result'):
                                print(f"  Duration: {status['result'].get('duration', 0):.2f}s")
                                print(f"  RTF: {status['result'].get('rtf', 0):.3f}")
                            break
                    else:
                        print(f"  Status check failed: {status_response.status_code}")

                print("\n📨 Check your webhook URL for:")
                print("  1. Transcription result (immediate)")
                print("  2. Quick intelligence (1-2s later)")
                print("  3. Enhanced intelligence (5-15s later)")

            else:
                print(f"❌ Error {response.status_code}: {response.text}")

            return result if response.status_code == 200 else None


async def test_intelligence_only():
    """Test intelligence extraction on existing transcript"""
    print("\n" + "="*60)
    print("Testing Intelligence-Only Extraction")
    print("="*60)

    test_transcript = """
    Hello everyone, thank you for joining today's quarterly sales review meeting.
    I'm pleased to report that we exceeded our Q3 targets by 15%, with total revenue
    reaching $2.5 million. Our new product line has been particularly successful.

    John from the sales team closed three major deals with Fortune 500 companies.
    Sarah will be following up with Acme Corp next week about their expansion plans.

    Action items for this quarter:
    1. Mike needs to prepare the Q4 forecast by Friday
    2. Lisa should schedule customer feedback sessions
    3. The marketing team must launch the holiday campaign by November 15th

    Overall, I'm very optimistic about our growth trajectory. Any questions?
    """

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Test sync intelligence extraction
        request_data = {
            "transcript_id": "test_123",
            "transcript_text": test_transcript,
            "mode": "sales"
        }
        headers = {"Authorization": f"Bearer {API_KEY}"}

        print("Sending transcript for intelligence extraction...")
        start_time = time.time()

        response = await client.post(
            f"{API_BASE_URL}/intelligence/extract/sync",
            json=request_data,
            headers=headers
        )

        elapsed = time.time() - start_time
        print(f"Response received in {elapsed:.2f}s")

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Intelligence Extraction Complete")
            print(f"Job ID: {result['job_id']}")
            print(f"Mode: {result.get('mode', 'N/A')}")
            print(f"Processing Time: {result.get('processing_time', 0):.2f}s")

            if result.get('intelligence'):
                intel = result['intelligence']
                print(f"\n📊 Extracted Intelligence:")
                print(f"  Call Type: {intel.get('call_type', 'N/A')}")
                print(f"  Intent: {intel.get('intent', 'N/A')}")
                print(f"  Sentiment: {intel.get('sentiment', 'N/A')}")

                if intel.get('summary'):
                    print(f"  Summary: {intel['summary'][:200]}...")

                if intel.get('action_items'):
                    print(f"\n  Action Items ({len(intel['action_items'])}):")
                    for item in intel['action_items'][:5]:
                        task = item.get('task', 'N/A')
                        assignee = item.get('assignee', 'Unassigned')
                        print(f"    - [{assignee}] {task}")

                if intel.get('people'):
                    print(f"\n  People Mentioned ({len(intel['people'])}):")
                    for person in intel['people'][:5]:
                        print(f"    - {person.get('name', 'N/A')} ({person.get('role', 'N/A')})")

                if intel.get('financial_info'):
                    fi = intel['financial_info']
                    if fi.get('amounts'):
                        print(f"\n  Financial Data:")
                        print(f"    Amounts: {fi['amounts']}")

        else:
            print(f"❌ Error {response.status_code}: {response.text}")

        return result if response.status_code == 200 else None


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("S2A Intelligence Integration Test Suite")
    print("="*60)

    # Check if test audio file exists
    if not Path(TEST_AUDIO_FILE).exists():
        print(f"\n❌ Test audio file not found: {TEST_AUDIO_FILE}")
        print("Please provide a test audio file (WAV format, 5 seconds to 2 minutes)")
        return

    try:
        # Test 1: Sync transcription with intelligence
        result1 = await test_sync_transcription_with_intelligence()

        # Small delay between tests
        await asyncio.sleep(2)

        # Test 2: Async transcription with intelligence
        result2 = await test_async_transcription_with_intelligence()

        # Small delay between tests
        await asyncio.sleep(2)

        # Test 3: Intelligence-only extraction
        result3 = await test_intelligence_only()

        # Summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"✅ Sync transcription with intelligence: {'PASSED' if result1 else 'FAILED'}")
        print(f"✅ Async transcription with intelligence: {'PASSED' if result2 else 'FAILED'}")
        print(f"✅ Intelligence-only extraction: {'PASSED' if result3 else 'FAILED'}")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())