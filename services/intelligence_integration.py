#!/usr/bin/env python3
"""
Intelligence Integration Service for S2A Pipeline
Orchestrates multi-stage intelligence extraction and webhook delivery
"""

import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from loguru import logger

from intelligence.legacy import ActionPipelineExtractor
from intelligence.intelligence_service import get_intelligence_service
from webhook import WebhookSender, WebhookPayload
from config import get_intelligence_settings


@dataclass
class IntelligenceJobRequest:
    job_id: str
    transcript: str
    callback_url: Optional[str] = None
    include_quick: bool = True
    include_enhanced: bool = True


class IntelligenceIntegrationService:
    """
    Orchestrates multi-stage intelligence processing:
    1. Quick intelligence (1-2 seconds) - immediate response
    2. Enhanced intelligence (5-15 seconds) - comprehensive analysis
    """

    def __init__(self):
        self.settings = get_intelligence_settings()
        self.webhook_sender = WebhookSender()
        self.quick_extractor = None

        # Initialize quick extractor
        if self.settings.enabled:
            try:
                self.quick_extractor = ActionPipelineExtractor(
                    base_url=self.settings.vllm_base_url
                )
                logger.info("Quick intelligence extractor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize quick extractor: {e}")

    async def process_intelligence_pipeline(self, request: IntelligenceJobRequest) -> Dict[str, Any]:
        """
        Process complete intelligence pipeline with multi-stage delivery

        Returns quick intelligence immediately, then processes enhanced intelligence
        in background with webhook delivery
        """

        results = {
            "job_id": request.job_id,
            "quick_intelligence": None,
            "enhanced_intelligence_submitted": False,
            "webhook_status": {}
        }

        # Stage 1: Quick Intelligence (if enabled and available)
        if request.include_quick and self.quick_extractor:
            try:
                logger.info(f"Processing quick intelligence for job {request.job_id}")
                quick_result = await self._extract_quick_intelligence(request.transcript)

                if quick_result["success"]:
                    results["quick_intelligence"] = quick_result
                    logger.info(f"Quick intelligence completed for job {request.job_id} in {quick_result['latency']:.2f}s")

                    # Send quick intelligence webhook if callback URL provided
                    if request.callback_url:
                        webhook_success = await self._send_intelligence_webhook(
                            request.job_id,
                            request.callback_url,
                            quick_result,
                            "quick"
                        )
                        results["webhook_status"]["quick"] = webhook_success

                else:
                    logger.warning(f"Quick intelligence failed for job {request.job_id}: {quick_result.get('error')}")

            except Exception as e:
                logger.error(f"Quick intelligence error for job {request.job_id}: {e}")

        # Stage 2: Enhanced Intelligence (background processing)
        if request.include_enhanced and self.settings.enabled:
            try:
                # Submit to enhanced intelligence service
                intelligence_service = await get_intelligence_service()

                enhanced_job_id = await intelligence_service.submit_job(
                    transcript_id=request.job_id,
                    transcript_text=request.transcript,
                    priority="normal"
                )

                results["enhanced_intelligence_submitted"] = True
                results["enhanced_job_id"] = enhanced_job_id

                logger.info(f"Enhanced intelligence job {enhanced_job_id} submitted for {request.job_id}")

                # Schedule background webhook delivery for enhanced intelligence
                if request.callback_url:
                    asyncio.create_task(
                        self._monitor_enhanced_intelligence(
                            enhanced_job_id,
                            request.job_id,
                            request.callback_url
                        )
                    )

            except Exception as e:
                logger.error(f"Enhanced intelligence submission failed for job {request.job_id}: {e}")

        return results

    async def _extract_quick_intelligence(self, transcript: str) -> Dict[str, Any]:
        """Extract quick intelligence using legacy extractor"""
        loop = asyncio.get_event_loop()

        # Run in thread pool to avoid blocking
        return await loop.run_in_executor(
            None,
            self.quick_extractor.extract_quick,
            transcript
        )

    async def _send_intelligence_webhook(self,
                                       job_id: str,
                                       callback_url: str,
                                       intelligence_result: Dict[str, Any],
                                       intelligence_type: str) -> bool:
        """Send intelligence results via webhook"""

        try:
            payload = WebhookPayload(
                job_id=job_id,
                status="completed" if intelligence_result["success"] else "failed",
                intelligence_type=intelligence_type,
                intelligence_data=intelligence_result.get("data"),
                processing_time=intelligence_result.get("latency"),
                error=intelligence_result.get("error")
            )

            success = await self.webhook_sender.send_webhook(callback_url, payload)

            if success:
                logger.info(f"Sent {intelligence_type} intelligence webhook for job {job_id}")
            else:
                logger.warning(f"Failed to send {intelligence_type} intelligence webhook for job {job_id}")

            return success

        except Exception as e:
            logger.error(f"Error sending {intelligence_type} intelligence webhook for job {job_id}: {e}")
            return False

    async def _monitor_enhanced_intelligence(self,
                                           enhanced_job_id: str,
                                           original_job_id: str,
                                           callback_url: str):
        """Monitor enhanced intelligence job and send webhook when complete"""

        max_wait_time = 300  # 5 minutes max wait
        check_interval = 5   # Check every 5 seconds
        start_time = time.time()

        try:
            intelligence_service = await get_intelligence_service()

            while time.time() - start_time < max_wait_time:
                # Check job status
                status = await intelligence_service.get_job_status(enhanced_job_id)

                if status and status["status"] in ["completed", "failed"]:
                    # Get result
                    result = await intelligence_service.get_job_result(enhanced_job_id)

                    if result:
                        # Send enhanced intelligence webhook
                        await self._send_intelligence_webhook(
                            original_job_id,
                            callback_url,
                            result,
                            "enhanced"
                        )
                    else:
                        logger.warning(f"No result found for enhanced intelligence job {enhanced_job_id}")

                    return

                # Wait before next check
                await asyncio.sleep(check_interval)

            # Timeout
            logger.warning(f"Enhanced intelligence job {enhanced_job_id} timed out after {max_wait_time}s")

            # Send timeout webhook
            timeout_payload = WebhookPayload(
                job_id=original_job_id,
                status="timeout",
                intelligence_type="enhanced",
                error="Enhanced intelligence processing timed out"
            )
            await self.webhook_sender.send_webhook(callback_url, timeout_payload)

        except Exception as e:
            logger.error(f"Error monitoring enhanced intelligence job {enhanced_job_id}: {e}")

    def close(self):
        """Clean up resources"""
        if self.quick_extractor:
            self.quick_extractor.close()


# Global intelligence integration service
_intelligence_integration: Optional[IntelligenceIntegrationService] = None


def get_intelligence_integration() -> IntelligenceIntegrationService:
    """Get global intelligence integration service instance"""
    global _intelligence_integration

    if _intelligence_integration is None:
        _intelligence_integration = IntelligenceIntegrationService()

    return _intelligence_integration


async def process_transcript_intelligence(job_id: str,
                                        transcript: str,
                                        callback_url: Optional[str] = None,
                                        include_quick: bool = True,
                                        include_enhanced: bool = True) -> Dict[str, Any]:
    """
    Convenience function to process transcript through intelligence pipeline

    Args:
        job_id: Unique identifier for the transcription job
        transcript: The transcription text
        callback_url: Optional webhook URL for intelligence results
        include_quick: Whether to include quick intelligence (default: True)
        include_enhanced: Whether to include enhanced intelligence (default: True)

    Returns:
        Dictionary with quick intelligence results and enhanced job status
    """

    service = get_intelligence_integration()

    request = IntelligenceJobRequest(
        job_id=job_id,
        transcript=transcript,
        callback_url=callback_url,
        include_quick=include_quick,
        include_enhanced=include_enhanced
    )

    return await service.process_intelligence_pipeline(request)