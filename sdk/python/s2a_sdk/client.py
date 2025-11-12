"""
S2A Python SDK Client Implementation
"""

import os
import asyncio
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, BinaryIO
from urllib.parse import urljoin

import httpx
from .models import (
    IntelligenceResult,
    QuickIntelligenceResult,
    CompleteResult,
    AsyncJob,
    JobStatus,
    S2AError,
    AudioValidationError,
    RateLimitError,
    AuthenticationError,
    TimeoutError,
    IntelligenceUnavailableError
)
from .enums import (
    JobStatusType,
    IntelligenceMode,
    Priority,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    MIN_ASYNC_AUDIO_DURATION,
    MAX_ASYNC_AUDIO_DURATION,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
)
from .utils import AudioValidator, parse_response, retry_with_backoff


class S2AClient:
    """
    Main S2A SDK client for speech-to-text and intelligence extraction

    Example:
        ```python
        client = S2AClient(api_key="bp-proj-your-key")

        # Simple transcription
        result = client.transcribe("audio.wav")

        # Async transcription with intelligence
        job = client.transcribe_async_with_intelligence(
            "long_meeting.mp3",
            callback_url="https://yourapp.com/webhook"
        )
        ```
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY
    ):
        """
        Initialize S2A client

        Args:
            api_key: S2A API key (format: bp-proj-*, bp-*, or bp-svc-*)
            base_url: Base URL for S2A API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
        """
        if not api_key:
            raise ValueError("API key is required")

        if not api_key.startswith(('bp-proj-', 'bp-', 'bp-svc-')):
            raise ValueError("Invalid API key format. Must start with bp-proj-, bp-, or bp-svc-")

        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize HTTP client
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"S2A-Python-SDK/1.0.0"
            }
        )

        # Initialize async client for concurrent operations
        self._async_client = None

        # Audio validator
        self._audio_validator = AudioValidator()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    def close(self):
        """Close HTTP client"""
        if self._client:
            self._client.close()

    async def aclose(self):
        """Close async HTTP client"""
        if self._async_client:
            await self._async_client.aclose()
        self.close()

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": f"S2A-Python-SDK/1.0.0"
                }
            )
        return self._async_client

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and convert errors"""
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Invalid API key or insufficient permissions")
            elif e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                raise RateLimitError("Rate limit exceeded", retry_after=retry_after)
            elif e.response.status_code == 413:
                raise AudioValidationError("Audio file too large")
            elif e.response.status_code == 422:
                try:
                    error_data = e.response.json()
                    raise AudioValidationError(f"Audio validation failed: {error_data.get('detail', 'Unknown error')}")
                except ValueError:
                    raise AudioValidationError("Audio validation failed")
            else:
                try:
                    error_data = e.response.json()
                    raise S2AError(
                        error_data.get("detail", f"HTTP {e.response.status_code} error"),
                        status_code=e.response.status_code,
                        response_data=error_data
                    )
                except ValueError:
                    raise S2AError(f"HTTP {e.response.status_code} error", status_code=e.response.status_code)

    # Core Transcription Methods
    @retry_with_backoff()
    def transcribe_async(
        self,
        audio_file: Union[str, Path, BinaryIO],
        callback_url: str,
        enhance_audio: bool = True,
        remove_silence: bool = False,
        priority: Priority = Priority.NORMAL
    ) -> AsyncJob:
        """
        Asynchronous audio transcription (min 1 sec and max 5 hours)

        Args:
            audio_file: Path to audio file or file-like object
            callback_url: URL to receive completion webhook
            enhance_audio: Apply audio enhancement
            remove_silence: Remove silent portions
            priority: Processing priority

        Returns:
            AsyncJob with job ID and status

        Raises:
            AudioValidationError: Invalid audio file or too long
            AuthenticationError: Invalid API key
            RateLimitError: API rate limit exceeded
        """
        # Validate audio file
        file_path, file_obj = self._audio_validator.prepare_audio_file(audio_file)
        duration = self._audio_validator.get_audio_duration(file_path or audio_file)

        if duration < MIN_ASYNC_AUDIO_DURATION:
            raise AudioValidationError(
                f"Audio duration ({duration:.1f}s) is less than async API limit ({MIN_ASYNC_AUDIO_DURATION}s)."
            )
        if duration > MAX_ASYNC_AUDIO_DURATION:
            raise AudioValidationError(
                f"Audio duration ({duration:.1f}s) exceeds async API limit ({MAX_ASYNC_AUDIO_DURATION}s)."
            )

        # Prepare request
        files = {"audio_file": file_obj}
        data = {
            "callback_url": callback_url,
            "enhance_audio": enhance_audio,
            "remove_silence": remove_silence,
            "priority": priority.value
        }

        # Make request
        response = self._client.post(
            f"{self.base_url}/v1/transcribe",
            files=files,
            data=data
        )

        result_data = self._handle_response(response)
        return parse_response(result_data, AsyncJob)

    # Intelligence Methods
    @retry_with_backoff()
    def extract_intelligence(
        self,
        transcript: str,
        mode: IntelligenceMode = IntelligenceMode.AUTO_DETECT
    ) -> IntelligenceResult:
        """
        Extract comprehensive business intelligence from transcript

        Args:
            transcript: Transcription text
            mode: Intelligence extraction mode

        Returns:
            IntelligenceResult with comprehensive analysis

        Raises:
            IntelligenceUnavailableError: Intelligence service unavailable
        """
        data = {
            "transcript_id": f"sdk_{int(time.time())}",
            "transcript_text": transcript,
            "mode": mode.value
        }

        try:
            response = self._client.post(
                f"{self.base_url}/v1/intelligence/extract/sync",
                json=data
            )
            result_data = self._handle_response(response)
            return parse_response(result_data["intelligence"], IntelligenceResult)
        except S2AError as e:
            if e.status_code in [503, 502]:
                raise IntelligenceUnavailableError("Intelligence service temporarily unavailable")
            raise

    @retry_with_backoff()
    def extract_quick_intelligence(self, transcript: str) -> QuickIntelligenceResult:
        """
        Extract quick intelligence insights (1-2 seconds)

        Args:
            transcript: Transcription text

        Returns:
            QuickIntelligenceResult with basic insights
        """
        # Use legacy quick extraction endpoint
        data = {
            "transcript_id": f"sdk_quick_{int(time.time())}",
            "transcript_text": transcript,
            "mode": IntelligenceMode.QUICK.value
        }

        response = self._client.post(
            f"{self.base_url}/v1/intelligence/extract/sync",
            json=data
        )

        result_data = self._handle_response(response)
        return parse_response(result_data["intelligence"], QuickIntelligenceResult)

    def transcribe_async_with_intelligence(
        self,
        audio_file: Union[str, Path, BinaryIO],
        callback_url: str,
        intelligence_mode: IntelligenceMode = IntelligenceMode.AUTO_DETECT,
        enhance_audio: bool = True,
        priority: Priority = Priority.NORMAL,
        include_intelligence: bool = True
    ) -> AsyncJob:
        """
        Asynchronous transcription with automatic intelligence extraction

        Args:
            audio_file: Path to audio file or file-like object
            callback_url: URL to receive completion webhook
            intelligence_mode: Intelligence extraction mode
            enhance_audio: Apply audio enhancement
            priority: Processing priority
            include_intelligence: Include intelligence in results

        Returns:
            AsyncJob with job ID and status

        Note:
            When complete, webhook will include both transcription and intelligence data
        """
        # For async, the server will handle intelligence automatically
        # We add intelligence parameters to the callback URL
        enhanced_callback = f"{callback_url}?intelligence_mode={intelligence_mode.value}&include_intelligence={include_intelligence}"

        return self.transcribe_async(
            audio_file=audio_file,
            callback_url=enhanced_callback,
            enhance_audio=enhance_audio,
            priority=priority
        )

    # Job Management
    @retry_with_backoff()
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of async job"""
        response = self._client.get(f"{self.base_url}/v1/transcribe/status/{job_id}")
        result_data = self._handle_response(response)
        return parse_response(result_data, JobStatus)

    def wait_for_completion(
        self,
        job_id: str,
        timeout: Optional[float] = None,
        poll_interval: float = 5.0
    ) -> CompleteResult:
        """
        Wait for async job completion and return results

        Args:
            job_id: Job ID from async transcription
            timeout: Maximum wait time in seconds
            poll_interval: How often to check status

        Returns:
            CompleteResult when job completes

        Raises:
            TimeoutError: Job didn't complete within timeout
        """
        start_time = time.time()
        timeout = timeout or self.timeout

        while True:
            status = self.get_job_status(job_id)

            if status.status == JobStatusType.COMPLETED:
                # Get the full result
                response = self._client.get(f"{self.base_url}/v1/transcription/result/{job_id}")
                result_data = self._handle_response(response)
                return parse_response(result_data, CompleteResult)

            elif status.status == JobStatusType.FAILED:
                raise S2AError(f"Job failed: {status.error_message}")

            elif time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

            time.sleep(poll_interval)

    # Utility Methods
    def validate_audio(self, audio_file: Union[str, Path, BinaryIO]) -> Dict[str, Any]:
        """
        Validate audio file without processing

        Args:
            audio_file: Path to audio file or file-like object

        Returns:
            Dict with validation results and metadata
        """
        return self._audio_validator.validate_audio_file(audio_file)

    def estimate_cost(self, duration_seconds: float) -> Dict[str, Any]:
        """
        Estimate processing cost for audio duration

        Args:
            duration_seconds: Audio duration in seconds

        Returns:
            Dict with cost estimation
        """
        # This would integrate with actual pricing API
        return {
            "duration_seconds": duration_seconds,
            "estimated_processing_time": duration_seconds * 0.1,  # RTF of 0.1
            "tier": "standard"
        }

    def health_check(self) -> Dict[str, Any]:
        """Check API health and connectivity"""
        try:
            response = self._client.get(f"{self.base_url}/v1/statistics/health")
            return self._handle_response(response)
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}