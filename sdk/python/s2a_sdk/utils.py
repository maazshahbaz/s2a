"""
Utility functions for S2A SDK
"""

import os
import time
import mimetypes
from pathlib import Path
from typing import Union, Dict, Any, BinaryIO, Tuple, Optional, Type, TypeVar
from functools import wraps
import tempfile

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from .enums import (
    MAX_FILE_SIZE,
    MIN_AUDIO_DURATION,
    SUPPORTED_MIME_TYPES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RETRY_BACKOFF
)
from .models import S2AError, AudioValidationError

T = TypeVar('T')


def retry_with_backoff(max_retries: int = 3, delay: float = DEFAULT_RETRY_DELAY, backoff: float = DEFAULT_RETRY_BACKOFF):
    """
    Decorator for retrying functions with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Don't retry certain error types
                    if isinstance(e, (AudioValidationError, ValueError)):
                        raise

                    if attempt < max_retries:
                        sleep_time = delay * (backoff ** attempt)
                        time.sleep(sleep_time)
                    else:
                        raise last_exception

            raise last_exception
        return wrapper
    return decorator


def parse_response(data: Dict[str, Any], model_class: Type[T]) -> T:
    """
    Parse API response data into model object

    Args:
        data: Response data dictionary
        model_class: Target model class

    Returns:
        Instance of model_class
    """
    try:
        # Handle nested data structures
        if hasattr(model_class, '__annotations__'):
            # Convert snake_case to match model field names
            converted_data = {}
            for key, value in data.items():
                # Handle common API response field mappings
                converted_key = key
                if key == 'job_id':
                    converted_key = 'job_id'
                elif key == 'processing_time':
                    converted_key = 'processing_time'

                converted_data[converted_key] = value

            return model_class(**converted_data)
        else:
            return model_class(**data)

    except TypeError as e:
        raise S2AError(f"Failed to parse response data: {e}")


class AudioValidator:
    """Validates audio files before processing"""

    def __init__(self):
        self.supported_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.mp4'}

    def validate_audio_file(self, audio_file: Union[str, Path, BinaryIO]) -> Dict[str, Any]:
        """
        Validate audio file and return metadata

        Args:
            audio_file: Path to audio file or file-like object

        Returns:
            Dict with validation results and metadata

        Raises:
            AudioValidationError: If file is invalid
        """
        if isinstance(audio_file, (str, Path)):
            return self._validate_file_path(Path(audio_file))
        else:
            return self._validate_file_object(audio_file)

    def _validate_file_path(self, file_path: Path) -> Dict[str, Any]:
        """Validate file at given path"""
        if not file_path.exists():
            raise AudioValidationError(f"Audio file not found: {file_path}")

        if not file_path.is_file():
            raise AudioValidationError(f"Path is not a file: {file_path}")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise AudioValidationError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB "
                f"(max: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB)"
            )

        if file_size == 0:
            raise AudioValidationError("Audio file is empty")

        # Check extension
        if file_path.suffix.lower() not in self.supported_extensions:
            raise AudioValidationError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self.supported_extensions)}"
            )

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type not in SUPPORTED_MIME_TYPES:
            # Try to determine from extension
            ext_to_mime = {
                '.wav': 'audio/wav',
                '.mp3': 'audio/mpeg',
                '.flac': 'audio/flac',
                '.m4a': 'audio/m4a',
                '.ogg': 'audio/ogg',
                '.mp4': 'video/mp4'
            }
            mime_type = ext_to_mime.get(file_path.suffix.lower(), 'audio/wav')

        # Get audio duration if possible
        duration = self.get_audio_duration(file_path)

        return {
            "valid": True,
            "file_size": file_size,
            "mime_type": mime_type,
            "duration": duration,
            "format": SUPPORTED_MIME_TYPES.get(mime_type, "unknown")
        }

    def _validate_file_object(self, file_obj: BinaryIO) -> Dict[str, Any]:
        """Validate file-like object"""
        if not hasattr(file_obj, 'read'):
            raise AudioValidationError("Invalid file object: missing read method")

        # Try to get file size
        current_pos = file_obj.tell() if hasattr(file_obj, 'tell') else 0

        try:
            if hasattr(file_obj, 'seek') and hasattr(file_obj, 'tell'):
                file_obj.seek(0, 2)  # Seek to end
                file_size = file_obj.tell()
                file_obj.seek(current_pos)  # Restore position

                if file_size > MAX_FILE_SIZE:
                    raise AudioValidationError(
                        f"File too large: {file_size / 1024 / 1024:.1f}MB "
                        f"(max: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB)"
                    )
            else:
                file_size = None
        except (OSError, IOError):
            file_size = None

        return {
            "valid": True,
            "file_size": file_size,
            "mime_type": "application/octet-stream",  # Generic for file objects
            "duration": None,  # Cannot determine without saving to disk
            "format": "unknown"
        }

    def get_audio_duration(self, audio_file: Union[str, Path]) -> Optional[float]:
        """
        Get audio duration in seconds

        Args:
            audio_file: Path to audio file

        Returns:
            Duration in seconds or None if cannot determine
        """
        if not HAS_LIBROSA:
            # Fallback: try to use ffmpeg or other methods
            return self._get_duration_fallback(audio_file)

        try:
            duration = librosa.get_duration(filename=str(audio_file))

            if duration < MIN_AUDIO_DURATION:
                raise AudioValidationError(
                    f"Audio too short: {duration:.1f}s (minimum: {MIN_AUDIO_DURATION}s)"
                )

            return duration

        except Exception as e:
            # If librosa fails, try fallback methods
            return self._get_duration_fallback(audio_file)

    def _get_duration_fallback(self, audio_file: Union[str, Path]) -> Optional[float]:
        """Fallback method to get duration without librosa"""
        try:
            import subprocess
            import json

            # Try ffprobe first
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', str(audio_file)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data['format']['duration'])
                return duration

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
                FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
            pass

        # If all methods fail, return None
        return None

    def prepare_audio_file(self, audio_file: Union[str, Path, BinaryIO]) -> Tuple[Optional[Path], BinaryIO]:
        """
        Prepare audio file for upload

        Args:
            audio_file: Path to audio file or file-like object

        Returns:
            Tuple of (file_path, file_object) ready for upload
        """
        if isinstance(audio_file, (str, Path)):
            file_path = Path(audio_file)
            self.validate_audio_file(file_path)
            file_obj = open(file_path, 'rb')
            return file_path, file_obj
        else:
            # File-like object
            self.validate_audio_file(audio_file)
            return None, audio_file


class WebhookValidator:
    """Validates webhook URLs and signatures"""

    @staticmethod
    def validate_webhook_url(url: str) -> bool:
        """
        Validate webhook URL format

        Args:
            url: Webhook URL

        Returns:
            True if valid

        Raises:
            ValueError: If URL is invalid
        """
        if not url:
            raise ValueError("Webhook URL is required")

        if not url.startswith(('http://', 'https://')):
            raise ValueError("Webhook URL must start with http:// or https://")

        # Basic URL validation
        from urllib.parse import urlparse
        parsed = urlparse(url)

        if not parsed.netloc:
            raise ValueError("Invalid webhook URL: missing hostname")

        return True


class ResponseParser:
    """Parses different types of API responses"""

    @staticmethod
    def parse_transcription_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse transcription API response"""
        return {
            "job_id": data.get("job_id"),
            "text": data.get("text", ""),
            "duration": data.get("duration", 0.0),
            "confidence": data.get("confidence", 0.0),
            "processing_time": data.get("processing_time", 0.0),
            "rtf": data.get("rtf", 0.0),
            "chunks": data.get("chunks", 1),
            "audio_quality": data.get("audio_quality")
        }

    @staticmethod
    def parse_intelligence_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse intelligence API response"""
        intelligence_data = data.get("intelligence", {})

        # Handle different intelligence response formats
        if "quick_intelligence" in data:
            return ResponseParser._parse_quick_intelligence(data["quick_intelligence"])
        elif "enhanced_intelligence" in data:
            return ResponseParser._parse_enhanced_intelligence(data["enhanced_intelligence"])
        else:
            return intelligence_data

    @staticmethod
    def _parse_quick_intelligence(data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse quick intelligence response"""
        return {
            "summary": data.get("summary", ""),
            "intent": data.get("intent", "general_discussion"),
            "sentiment": data.get("sentiment", "neutral"),
            "action_items": data.get("action_items", []),
            "key_entities": data.get("key_entities", []),
            "confidence_score": data.get("confidence_score", 0.8),
            "processing_time": data.get("processing_time", 0.0)
        }

    @staticmethod
    def _parse_enhanced_intelligence(data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse enhanced intelligence response"""
        return {
            "call_type": data.get("call_type", "internal_meeting"),
            "intent": data.get("intent", "general_discussion"),
            "sentiment": data.get("sentiment", "neutral"),
            "summary": data.get("summary", ""),
            "key_topics": data.get("key_topics", []),
            "people": data.get("entities", {}).get("people", []),
            "companies": data.get("entities", {}).get("companies", []),
            "products": data.get("entities", {}).get("products", []),
            "action_items": data.get("action_items", []),
            "emails": data.get("entities", {}).get("emails", []),
            "phones": data.get("entities", {}).get("phones", []),
            "dates": data.get("entities", {}).get("dates", []),
            "financial_info": data.get("entities", {}).get("financial_info", {}),
            "opportunity_info": data.get("opportunity_info"),
            "issues": data.get("issues", []),
            "conversation_metrics": data.get("conversation_metrics", {}),
            "confidence_score": data.get("confidence_score", 0.8),
            "completeness_score": data.get("completeness_score", 0.8),
            "recommendations": data.get("recommendations", []),
            "risk_flags": data.get("risk_flags", [])
        }


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_file_size(bytes_size: int) -> str:
    """Format file size in bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"