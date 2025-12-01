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
    timestamp: float = None
    processing_time: Optional[float] = None
    
    # Intelligence-specific fields
    intelligence_type: Optional[str] = None  # "quick", "enhanced", "transcription"
    intelligence_data: Optional[Dict[str, Any]] = None
    
    # Additional fields for backward compatibility
    transcription: Optional[str] = None
    ai_analysis: Optional[Dict[str, Any]] = None
    diarized_transcription: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class WebhookSender:
    def __init__(self, timeout: float = 30.0, max_retries: int = 3, retry_delay: float = 1.0):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def validate_callback_url(self, url: str) -> bool:
        """Validate callback URL format and security"""
        try:
            parsed = urlparse(url)
            
            # Must be HTTP or HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False
                
            # Must have a valid hostname
            if not parsed.netloc:
                return False
                
            # Block localhost/private IPs for security (optional - remove if needed)
            # This prevents SSRF attacks
            hostname = parsed.hostname
            if hostname and (
                hostname in ['localhost', '127.0.0.1', '::1'] or
                hostname.startswith('10.') or
                hostname.startswith('192.168.') or
                hostname.startswith('172.')
            ):
                logger.warning(f"Blocked private/localhost callback URL: {url}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Invalid callback URL format: {url}, error: {e}")
            return False
    
    async def send_webhook(self, callback_url: str, payload: WebhookPayload) -> bool:
        """Send webhook with retry logic"""
        
        if not self.validate_callback_url(callback_url):
            logger.error(f"Invalid callback URL: {callback_url}")
            return False
        
        # Prepare payload
        webhook_data = {
            "job_id": payload.job_id,
            "status": payload.status,
            "timestamp": payload.timestamp,
            "processing_time": payload.processing_time
        }

        # Add intelligence fields if present
        if payload.intelligence_type:
            webhook_data["intelligence_type"] = payload.intelligence_type

        if payload.intelligence_data:
            webhook_data["intelligence_data"] = payload.intelligence_data

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