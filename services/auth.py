"""
OpenAI-style API Key Authentication for FastAPI
Features:
- Secure API key storage (hash only)
- Bearer token auth
- Rate limiting (minute/hour/day)
- Permission-based access control
- File-based persistence (atomic writes)
- Extensible for Redis or DB backend
"""

import hashlib
import hmac
import secrets
import time
import json
import os
from tempfile import NamedTemporaryFile
from typing import Optional, Dict, List
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from fastapi import HTTPException, Request, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from loguru import logger

# ======================================================
# CONFIGURATION
# ======================================================

API_KEY_SECRET = os.environ.get("API_KEY_SECRET", "change-me")  # Use env var in production
API_KEYS_FILE = "./api_keys.json"


# ======================================================
# UTILITY FUNCTIONS
# ======================================================

def hash_api_key(api_key: str) -> str:
    """HMAC SHA-256 hash of API key (secure, secret-based)"""
    return hmac.new(API_KEY_SECRET.encode(), api_key.encode(), hashlib.sha256).hexdigest()


# ======================================================
# ENUMS & MODELS
# ======================================================

class APIKeyType(Enum):
    """API Key types similar to OpenAI"""
    PROJECT = "bp-proj"
    USER = "bp"
    SERVICE = "bp-svc"


class APIKey(BaseModel):
    key_id: str
    key_hash: str
    name: str
    key_type: APIKeyType
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_active: bool = True

    # Rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000

    # Permissions
    permissions: List[str] = Field(default_factory=lambda: ["transcribe", "status", "stats"])

    # Usage stats
    total_audio_minutes: float = 0.0
    total_requests: int = 0


class RateLimitInfo(BaseModel):
    allowed: bool
    requests_remaining_minute: int
    requests_remaining_hour: int
    requests_remaining_day: int
    reset_time_minute: int
    reset_time_hour: int
    reset_time_day: int


# ======================================================
# API KEY STORE (FILE-BASED)
# ======================================================

class APIKeyStore:
    """File-based API key storage (use Redis/DB for production scaling)"""

    def __init__(self, storage_path: str = API_KEYS_FILE):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: Dict[str, APIKey] = {}
        self._load_keys()

    def _load_keys(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for key_hash, key_data in data.items():
                        kt = key_data.pop('key_type')
                        created = key_data.pop('created_at')
                        last_used = key_data.pop('last_used')
                        self._keys[key_hash] = APIKey(
                            **key_data,
                            key_type=APIKeyType(kt),
                            created_at=datetime.fromisoformat(created),
                            last_used=datetime.fromisoformat(last_used) if last_used else None
                        )
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")

    def _save_keys(self):
        try:
            data = {}
            for key_hash, key_info in self._keys.items():
                data[key_hash] = {
                    **key_info.dict(),
                    'key_type': key_info.key_type.value,
                    'created_at': key_info.created_at.isoformat(),
                    'last_used': key_info.last_used.isoformat() if key_info.last_used else None
                }
            # Atomic write
            with NamedTemporaryFile('w', delete=False, dir=str(self.storage_path.parent)) as tf:
                tmpname = tf.name
                json.dump(data, tf, indent=2)
            os.chmod(tmpname, 0o600)
            os.replace(tmpname, self.storage_path)
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")

    def create_key(self, name: str, key_type: APIKeyType = APIKeyType.PROJECT, **kwargs) -> tuple[str, APIKey]:
        # Generate plaintext key
        key_suffix = secrets.token_urlsafe(32)
        api_key = f"{key_type.value}-{key_suffix}"
        key_hash = hash_api_key(api_key)

        key_info = APIKey(
            key_id=key_hash[:12],
            key_hash=key_hash,
            name=name,
            key_type=key_type,
            created_at=datetime.now(timezone.utc),
            **kwargs
        )

        self._keys[key_hash] = key_info
        self._save_keys()
        logger.info(f"Created API key: {key_info.key_id} ({name})")
        return api_key, key_info

    def get_key(self, api_key: str) -> Optional[APIKey]:
        key_hash = hash_api_key(api_key)
        return self._keys.get(key_hash)

    def update_key_usage(self, api_key: str, audio_duration: float = 0.0):
        key_hash = hash_api_key(api_key)
        if key_hash in self._keys:
            key_info = self._keys[key_hash]
            key_info.last_used = datetime.now(timezone.utc)
            key_info.usage_count += 1
            key_info.total_requests += 1
            key_info.total_audio_minutes += audio_duration / 60.0
            self._save_keys()

    def revoke_key(self, api_key: str) -> bool:
        key_hash = hash_api_key(api_key)
        if key_hash in self._keys:
            self._keys[key_hash].is_active = False
            self._save_keys()
            logger.info(f"Revoked API key: {self._keys[key_hash].key_id}")
            return True
        return False

    def list_keys(self) -> List[APIKey]:
        return list(self._keys.values())


# ======================================================
# RATE LIMITER (IN-MEMORY)
# ======================================================

class RateLimiter:
    """Simple in-memory sliding window (use Redis for production)"""

    def __init__(self):
        self._usage: Dict[str, Dict[str, List[float]]] = {}

    def _cleanup_old_requests(self, key_id: str, current_time: float):
        if key_id not in self._usage:
            self._usage[key_id] = {'minute': [], 'hour': [], 'day': []}

        usage = self._usage[key_id]
        usage['minute'] = [t for t in usage['minute'] if current_time - t < 60]
        usage['hour'] = [t for t in usage['hour'] if current_time - t < 3600]
        usage['day'] = [t for t in usage['day'] if current_time - t < 86400]

    def check_rate_limit(self, key_info: APIKey) -> RateLimitInfo:
        current_time = time.time()
        self._cleanup_old_requests(key_info.key_id, current_time)
        usage = self._usage[key_info.key_id]

        minute_count = len(usage['minute'])
        hour_count = len(usage['hour'])
        day_count = len(usage['day'])

        allowed = (
            minute_count < key_info.requests_per_minute and
            hour_count < key_info.requests_per_hour and
            day_count < key_info.requests_per_day
        )

        if allowed:
            usage['minute'].append(current_time)
            usage['hour'].append(current_time)
            usage['day'].append(current_time)
            minute_count += 1
            hour_count += 1
            day_count += 1

        return RateLimitInfo(
            allowed=allowed,
            requests_remaining_minute=max(0, key_info.requests_per_minute - minute_count),
            requests_remaining_hour=max(0, key_info.requests_per_hour - hour_count),
            requests_remaining_day=max(0, key_info.requests_per_day - day_count),
            reset_time_minute=(int(current_time // 60) + 1) * 60,
            reset_time_hour=(int(current_time // 3600) + 1) * 3600,
            reset_time_day=(int(current_time // 86400) + 1) * 86400
        )


# ======================================================
# FASTAPI AUTH HANDLERS
# ======================================================

api_key_store = APIKeyStore()
rate_limiter = RateLimiter()


class BearerTokenAuth(HTTPBearer):
    """OpenAI-style Bearer token authentication"""

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> APIKey:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)

        if not credentials or not credentials.credentials:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")

        api_key = credentials.credentials

        if not any(api_key.startswith(kt.value) for kt in APIKeyType):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key format")

        key_info = api_key_store.get_key(api_key)
        if not key_info or not key_info.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API key")

        rate_info = rate_limiter.check_rate_limit(key_info)
        if not rate_info.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit-Minute": str(key_info.requests_per_minute),
                    "X-RateLimit-Limit-Hour": str(key_info.requests_per_hour),
                    "X-RateLimit-Limit-Day": str(key_info.requests_per_day),
                    "X-RateLimit-Remaining-Minute": str(rate_info.requests_remaining_minute),
                    "X-RateLimit-Remaining-Hour": str(rate_info.requests_remaining_hour),
                    "X-RateLimit-Remaining-Day": str(rate_info.requests_remaining_day),
                    "X-RateLimit-Reset-Minute": str(rate_info.reset_time_minute),
                    "X-RateLimit-Reset-Hour": str(rate_info.reset_time_hour),
                    "X-RateLimit-Reset-Day": str(rate_info.reset_time_day),
                }
            )

        request.state.api_key = api_key
        request.state.key_info = key_info
        request.state.rate_info = rate_info

        return key_info


auth = BearerTokenAuth()


def require_permission(permission: str):
    def check_permission(key_info: APIKey = Depends(auth)):
        if permission not in key_info.permissions:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission '{permission}' required")
        return key_info
    return check_permission


def update_usage(request: Request, audio_duration: float = 0.0):
    if hasattr(request.state, 'api_key'):
        api_key_store.update_key_usage(request.state.api_key, audio_duration)


def get_rate_limit_headers(request: Request) -> Dict[str, str]:
    if hasattr(request.state, 'rate_info') and hasattr(request.state, 'key_info'):
        rate_info = request.state.rate_info
        key_info = request.state.key_info
        return {
            "X-RateLimit-Limit-Minute": str(key_info.requests_per_minute),
            "X-RateLimit-Limit-Hour": str(key_info.requests_per_hour),
            "X-RateLimit-Limit-Day": str(key_info.requests_per_day),
            "X-RateLimit-Remaining-Minute": str(rate_info.requests_remaining_minute),
            "X-RateLimit-Remaining-Hour": str(rate_info.requests_remaining_hour),
            "X-RateLimit-Remaining-Day": str(rate_info.requests_remaining_day),
        }
    return {}
