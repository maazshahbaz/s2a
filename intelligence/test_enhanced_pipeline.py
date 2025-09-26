#!/usr/bin/env python3
"""
Test script for enhanced intelligence pipeline
Tests the comprehensive business intelligence extraction capabilities
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any

from enhanced_extractor import EnhancedExtractor, ExtractionMode
from intelligence_service import IntelligenceService


class IntelligencePipelineTester:
    """Test suite for the enhanced intelligence pipeline"""

    def __init__(self, vllm_base_url: str = "http://localhost:8000/v1"):
        self.vllm_base_url = vllm_base_url
        self.test_results = []

    def get_test_transcripts(self) -> Dict[str, str]:
        """Get sample transcripts for different scenarios"""
        return {
            "sales_call": """
[Sales Call - 2024-01-15 14:30]

Agent Sarah: Hi John, thanks for taking the time to speak with me today about our Enterprise CRM solution.

John Miller: Of course, Sarah. I'm the VP of Sales at TechCorp Inc., and we're really struggling with our current system. Our team of 25 sales reps is having issues with data entry and our reporting is terrible.

Agent Sarah: I completely understand those pain points. What's your current solution costing you monthly?

John Miller: We're paying about $500 per month for our current CRM, but frankly, it's not worth it. We're looking to budget around $800-1000 monthly for something better.

Agent Sarah: Perfect. Our Enterprise package is $1,200 monthly for 25 users, but I can offer a 15% discount for an annual commitment, bringing it to $1,020. It includes automated data entry, advanced analytics, and seamless Salesforce integration.

John Miller: That's interesting. We do use Salesforce for our marketing. What about implementation time?

Agent Sarah: Typically 2-3 weeks. I can also include our premium onboarding package worth $2,500 at no extra cost if we close this quarter.

John Miller: I'm interested, but I need to get approval from our CFO, Michael Chen. Can you send a detailed proposal to john.miller@techcorp.com and copy michael.chen@techcorp.com?

Agent Sarah: Absolutely! I'll include ROI projections showing potential 30% time savings. When would be good for a demo with your team?

John Miller: How about next Tuesday at 2 PM? We'll need to see the Salesforce integration in action.

Agent Sarah: Perfect. I'll send calendar invites and the proposal by end of day. Our main competitors like HubSpot charge 20% more for similar features.

John Miller: That's good to know. My main concerns are data security and user adoption - our reps aren't very tech-savvy.

Agent Sarah: We have SOC 2 compliance and our interface is designed for simplicity. I'll include training materials in the proposal.

John Miller: Sounds great. Let's schedule that demo and see how it goes.
""",

            "customer_support": """
[Customer Support Call - 2024-01-15 10:15]

Agent Mike: Thank you for calling TechSupport, this is Mike. I see you're calling about case #CS-2024-0156. How can I help you today?

Customer Lisa: Hi Mike, yes I'm Lisa Johnson from Marketing Solutions LLC. I'm having a critical issue with our billing system integration. It's been down for 2 hours and my team can't process any invoices.

Agent Mike: I'm sorry to hear about this urgent issue, Lisa. Let me pull up your account. I see you're on our Professional plan with integration to QuickBooks. Can you tell me what error messages you're seeing?

Customer Lisa: The error code is ERR_INT_401 and it says "Authentication failed for external service." This is really costing us money - we have $50,000 in pending invoices.

Agent Mike: I understand this is business-critical. The ERR_INT_401 typically indicates an API key issue with QuickBooks. When did this start happening?

Customer Lisa: Around 8 AM this morning. Nothing changed on our end. Is this a known issue?

Agent Mike: Let me check our system status... I see we had a scheduled maintenance last night that may have affected some integrations. I'm escalating this to our integration team immediately.

Customer Lisa: How long will this take? I need to report to my CEO at 2 PM about our invoice status.

Agent Mike: I'm marking this as Priority 1. Our integration specialist Sarah Williams will call you within 30 minutes. I'm also sending you a temporary workaround document to process critical invoices manually.

Customer Lisa: Okay, but this is the third time this month we've had integration issues. I'm considering switching providers if this keeps happening.

Agent Mike: I completely understand your frustration, Lisa. I'm adding a note for our account manager to reach out about service credits and improvements. Sarah will have a permanent fix ready soon.

Customer Lisa: My direct number is 555-789-1234. Please make sure Sarah calls me immediately.

Agent Mike: Absolutely. I'll monitor this personally and send you updates every 30 minutes until resolved. Is lisa.johnson@marketingsolutions.com still your preferred email?

Customer Lisa: Yes, that's correct. Thank you for the quick response, Mike.
""",

            "internal_meeting": """
[Project Planning Meeting - 2024-01-15 11:00]

Manager Tom: Alright team, let's kick off our Q1 project planning. We have three major initiatives to discuss today.

Developer Alice: Before we start, I want to flag that the API migration project is running behind. We originally estimated 6 weeks but we're looking at 8-10 weeks now.

Manager Tom: What's causing the delay, Alice?

Developer Alice: The legacy database has more dependencies than expected. We need additional time for data mapping and testing.

QA Lead Bob: From a testing perspective, we'll need at least 2 weeks for full regression testing once development is complete.

Manager Tom: Okay, so we're looking at a total of 12 weeks. That pushes our go-live date to April 15th instead of March 15th.

Product Manager Carol: That's problematic. We promised the client a March delivery, and they're already paying $25,000 monthly for the current system.

Developer Alice: I understand the pressure, but rushing this could cause data corruption. We're dealing with 500,000 customer records.

Manager Tom: Let's find a middle ground. Alice, can you prepare a phased rollout plan? We could migrate 25% of customers first as a pilot.

Developer Alice: That's possible. I'd need David from DevOps to help with the infrastructure scaling.

DevOps David: I can allocate 40 hours this sprint to set up staging environments. We'll need budget approval for additional AWS resources though.

Manager Tom: How much are we talking?

DevOps David: Approximately $3,000 monthly for the staging and testing environments during the migration period.

Product Manager Carol: That's within our contingency budget. I'll get approval from finance by Wednesday.

QA Lead Bob: I also need to bring in a contractor for API testing. Jennifer Rodriguez from TestingExperts quoted $150 per hour for a 3-week engagement.

Manager Tom: So we're looking at additional costs of $3,000 monthly plus $18,000 for the contractor. Let me summarize our action items...

Developer Alice: I'll deliver the phased migration plan by Friday, January 19th.

DevOps David: I'll submit the AWS resource request and have environments ready by January 22nd.

QA Lead Bob: I'll finalize the contract with Jennifer and start API testing as soon as the first phase is ready.

Product Manager Carol: I'll communicate the revised timeline to the client and get finance approval for additional costs.

Manager Tom: Perfect. Let's reconvene next Monday to review progress. Any other blockers?

Developer Alice: Just one - we need the client to provide test data by January 25th or we'll face another delay.

Product Manager Carol: I'll coordinate with their IT team this week. Meeting adjourned.
"""
        }

    async def test_extractor_modes(self):
        """Test the enhanced extractor with different modes"""
        print("Testing Enhanced Extractor Modes...")

        extractor = EnhancedExtractor()
        transcripts = self.get_test_transcripts()

        try:
            for transcript_name, transcript_text in transcripts.items():
                print(f"\n--- Testing {transcript_name} ---")

                # Test auto-detection
                result = extractor.extract(transcript_text, mode=ExtractionMode.AUTO_DETECT)

                print(f"Auto-detected mode: {result.get('mode', 'unknown')}")
                print(f"Success: {result.get('success', False)}")
                print(f"Latency: {result.get('latency', 0):.2f}s")

                if result.get('success'):
                    data = result['data']
                    print(f"Intent: {data.get('intent', 'N/A')}")
                    print(f"Sentiment: {data.get('sentiment', 'N/A')}")
                    print(f"Action Items: {len(data.get('action_items', []))}")
                    print(f"People Found: {len(data.get('entities', {}).get('people', []))}")
                    print(f"Financial Info: {bool(data.get('entities', {}).get('financial_info', {}))}")

                    # Show key extracted information
                    if data.get('entities', {}).get('people'):
                        print("Key People:")
                        for person in data['entities']['people'][:3]:
                            print(f"  - {person.get('name', 'Unknown')}: {person.get('role', 'N/A')}")

                    if data.get('action_items'):
                        print("Action Items:")
                        for item in data['action_items'][:3]:
                            print(f"  - {item.get('task', 'Unknown')} (Assignee: {item.get('assignee', 'N/A')})")

                self.test_results.append({
                    "test": f"extractor_{transcript_name}",
                    "success": result.get('success', False),
                    "mode": result.get('mode', 'unknown'),
                    "latency": result.get('latency', 0),
                    "error": result.get('error')
                })

        finally:
            extractor.close()

    async def test_intelligence_service(self):
        """Test the intelligence service with async processing"""
        print("\n\nTesting Intelligence Service...")

        service = IntelligenceService()
        transcripts = self.get_test_transcripts()

        try:
            await service.start()

            # Submit jobs
            job_ids = []
            for transcript_name, transcript_text in transcripts.items():
                job_id = await service.submit_job(
                    transcript_id=f"test_{transcript_name}_{int(time.time())}",
                    transcript_text=transcript_text,
                    mode=ExtractionMode.AUTO_DETECT,
                    priority="high"
                )
                job_ids.append((job_id, transcript_name))
                print(f"Submitted job {job_id} for {transcript_name}")

            # Wait for completion and check results
            print("\nWaiting for job completion...")
            max_wait = 60  # 60 seconds max wait

            for job_id, transcript_name in job_ids:
                start_wait = time.time()

                while time.time() - start_wait < max_wait:
                    status = await service.get_job_status(job_id)

                    if status and status['status'] in ['completed', 'failed']:
                        print(f"\nJob {job_id} ({transcript_name}): {status['status']}")

                        if status['status'] == 'completed':
                            result = await service.get_job_result(job_id)
                            if result:
                                print(f"  Processing time: {status.get('processing_time', 0):.2f}s")
                                print(f"  Mode detected: {result.get('mode', 'unknown')}")

                                data = result.get('data', {})
                                print(f"  Entities found: {len(data.get('entities', {}).get('people', []))} people")
                                print(f"  Action items: {len(data.get('action_items', []))}")
                        else:
                            print(f"  Error: {status.get('error', 'Unknown error')}")

                        self.test_results.append({
                            "test": f"service_{transcript_name}",
                            "success": status['status'] == 'completed',
                            "processing_time": status.get('processing_time', 0),
                            "error": status.get('error')
                        })
                        break

                    await asyncio.sleep(2)
                else:
                    print(f"Job {job_id} timed out")
                    self.test_results.append({
                        "test": f"service_{transcript_name}",
                        "success": False,
                        "error": "Timeout"
                    })

            # Get service metrics
            metrics = await service.get_metrics()
            print(f"\nService Metrics:")
            print(f"  Total jobs: {metrics['total_jobs_processed']}")
            print(f"  Success rate: {metrics['successful_extractions']}/{metrics['total_jobs_processed']}")
            print(f"  Average latency: {metrics['average_processing_time']:.2f}s")

        finally:
            await service.stop()

    def generate_test_report(self):
        """Generate a comprehensive test report"""
        print("\n" + "="*50)
        print("ENHANCED INTELLIGENCE PIPELINE TEST REPORT")
        print("="*50)

        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result['success'])

        print(f"\nOverall Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Successful: {successful_tests}")
        print(f"  Failed: {total_tests - successful_tests}")
        print(f"  Success Rate: {(successful_tests/total_tests)*100:.1f}%")

        print(f"\nDetailed Results:")
        for result in self.test_results:
            status = "✓" if result['success'] else "✗"
            test_name = result['test']

            print(f"  {status} {test_name}")
            if not result['success'] and result.get('error'):
                print(f"    Error: {result['error']}")
            if result.get('latency'):
                print(f"    Latency: {result['latency']:.2f}s")
            if result.get('processing_time'):
                print(f"    Processing Time: {result['processing_time']:.2f}s")
            if result.get('mode'):
                print(f"    Mode: {result['mode']}")

        # Performance analysis
        latencies = [r['latency'] for r in self.test_results if r.get('latency')]
        if latencies:
            print(f"\nPerformance Analysis:")
            print(f"  Average Latency: {sum(latencies)/len(latencies):.2f}s")
            print(f"  Min Latency: {min(latencies):.2f}s")
            print(f"  Max Latency: {max(latencies):.2f}s")

        print(f"\nTest completed at: {datetime.now().isoformat()}")


async def main():
    """Run the comprehensive test suite"""
    print("Starting Enhanced Intelligence Pipeline Tests...")
    print("Make sure vLLM server is running on http://localhost:8000/v1")

    tester = IntelligencePipelineTester()

    try:
        # Test the enhanced extractor
        await tester.test_extractor_modes()

        # Test the intelligence service
        await tester.test_intelligence_service()

        # Generate report
        tester.generate_test_report()

    except Exception as e:
        print(f"Test suite failed: {e}")
        print("\nMake sure to start vLLM server first:")
        print("python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-model-len 16000 --trust-remote-code")


if __name__ == "__main__":
    asyncio.run(main())