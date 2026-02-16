"""
Webhook utility for sending transcription results to callback URLs
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional
from loguru import logger
import time
from dataclasses import dataclass
from urllib.parse import urlparse

@dataclass
class WebhookPayload:
    job_id: str
    status: Optional[str] = None
    error: Optional[str] = None
    transcription: Optional[str] = None
    ai_analysis: Optional[Dict[str, Any]] = None
    diarized_transcription: Optional[str] = None
    agent_tasks: Optional[Dict[str, Any]] = None


class WebhookSender:
    def __init__(self, timeout: float = 30.0, max_retries: int = 3, retry_delay: float = 1.0):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    def sanitize_url(self, url: str) -> str:
        """Sanitize URL by stripping whitespace, quotes, and semicolons"""
        if not url:
            return ""
        return url.strip("'\"; ")

    def validate_callback_url(self, url: str) -> bool:
        """Validate callback URL format and security"""
        if not url:
            logger.warning("Empty callback URL provided")
            return False
            
        try:
            # Clean the URL
            clean_url = self.sanitize_url(url)
            parsed = urlparse(clean_url)
            
            # Must be HTTP or HTTPS
            if parsed.scheme not in ['http', 'https']:
                logger.warning(f"Invalid callback URL scheme: {clean_url} (scheme: {parsed.scheme})")
                return False
                
            # Must have a valid hostname (netloc)
            if not parsed.netloc:
                logger.warning(f"Invalid callback URL host: {clean_url}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error parsing callback URL: {url}, error: {e}")
            return False

    
    async def send_webhook(self, callback_url: str, payload: WebhookPayload) -> bool:
        """Send webhook with retry logic"""
        
        if not self.validate_callback_url(callback_url):
            logger.error(f"Invalid callback URL: {callback_url}")
            return False
        
        # Prepare payload
        webhook_data = {
            "job_id": payload.job_id,
        }

        if payload.status:
            webhook_data["status"] = payload.status

        # Add error if present
        if payload.error:
            webhook_data["error"] = payload.error
            
        # Add optional backward-compatible fields
        if payload.transcription:
            webhook_data["transcription"] = payload.transcription
            
        if payload.ai_analysis:
            webhook_data["ai_analysis"] = payload.ai_analysis
            
        if payload.diarized_transcription:
            webhook_data["diarized_transcription"] = payload.diarized_transcription

        if payload.agent_tasks:
            webhook_data["agent_tasks"] = payload.agent_tasks

        # Send with retries
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post(
                        callback_url,
                        json=webhook_data,
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "BytePulse-S2A-Webhook/1.0"
                        }
                    ) as response:
                        if 200 <= response.status < 300:
                            logger.info(f"Webhook sent successfully to {callback_url} for job {payload.job_id}")
                            return True
                        else:
                            error_text = await response.text()
                            logger.warning(f"Webhook failed with status {response.status}: {error_text}")
                            last_error = f"HTTP {response.status}: {error_text}"
                            
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                logger.warning(f"Webhook timeout for {callback_url} (attempt {attempt + 1}/{self.max_retries})")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Webhook error for {callback_url} (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        logger.error(f"Webhook failed after {self.max_retries} attempts to {callback_url}: {last_error}")
        return False

# Global webhook sender instance
webhook_sender = WebhookSender()