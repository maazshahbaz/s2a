"""
OpenAI-style API Key Authentication for FastAPI
Features:
- Secure API key storage (hash only)
- Bearer token auth
- Rate limiting (minute/hour/day)
- Permission-based access control
- Prisma database persistence
- Production-ready with async support
"""

import hashlib
import hmac
import secrets
import time
import os
from typing import Optional, Dict, List
from datetime import datetime, timezone
from enum import Enum

from fastapi import HTTPException, Request, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from loguru import logger
from generated.prisma import Prisma

# ======================================================
# CONFIGURATION
# ======================================================

API_KEY_SECRET = os.environ.get("API_KEY_SECRET", "change-me")  # Use env var in production


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
    userId: int
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
# API KEY STORE (DATABASE-BASED)
# ======================================================

class PrismaAPIKeyStore:
    """Prisma database-based API key storage for production scaling"""

    def __init__(self, db: Prisma):
        self.db = db

    async def create_key(self, user_id:int, name: str, key_type: APIKeyType = APIKeyType.PROJECT, **kwargs) -> tuple[str, APIKey]:
        """Create a new API key and store it in the database"""
        # Generate plaintext key
        key_suffix = secrets.token_urlsafe(32)
        api_key = f"{key_type.value}-{key_suffix}"
        key_hash = hash_api_key(api_key)

        # Default permissions
        permissions = kwargs.get('permissions', ["transcribe", "status", "stats"])
        
        try:
            # Create in database
            auth_key = await self.db.authkey.create(
                data={
                    'hash': key_hash,
                    'userId':user_id,
                    'name': name,
                    'keyType': key_type.value,
                    'requestsPerMinute': kwargs.get('requests_per_minute', 60),
                    'requestsPerHour': kwargs.get('requests_per_hour', 1000),
                    'requestsPerDay': kwargs.get('requests_per_day', 10000),
                    'permissions': permissions,
                }
            )
            
            # Convert to APIKey model
            key_info = self._auth_key_to_api_key(auth_key)
            logger.info(f"Created API key: {key_info.key_id} ({name})")
            return api_key, key_info
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise HTTPException(status_code=500, detail="Failed to create API key")

    async def get_key(self, api_key: str) -> Optional[APIKey]:
        """Retrieve API key info from database"""
        key_hash = hash_api_key(api_key)
        
        try:
            auth_key = await self.db.authkey.find_unique(
                where={'hash': key_hash}
            )
            
            if auth_key:
                return self._auth_key_to_api_key(auth_key)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get API key: {e}")
            return None

    async def update_key_usage(self, api_key: str, audio_duration: float = 0.0, track_request: bool = True):
        """Update API key usage statistics
        
        Args:
            api_key: The API key to update
            audio_duration: Audio duration in seconds (optional)
            track_request: Whether to increment request counters (default: True)
        """
        key_hash = hash_api_key(api_key)
        
        try:
            # Build update data based on parameters
            update_data = {
                'lastUsed': datetime.now(timezone.utc),
            }
            
            # Only increment request counters if tracking a new request
            if track_request:
                update_data.update({
                    'usageCount': {'increment': 1},
                    'totalRequests': {'increment': 1},
                })
            
            # Always track audio duration if provided
            if audio_duration > 0:
                update_data['totalAudioMinutes'] = {'increment': audio_duration / 60.0}
            
            await self.db.authkey.update(
                where={'hash': key_hash},
                data=update_data
            )
        except Exception as e:
            logger.error(f"Failed to update API key usage: {e}")

    async def revoke_key(self, api_key: str) -> bool:
        """Revoke (deactivate) an API key"""
        key_hash = hash_api_key(api_key)
        
        try:
            result = await self.db.authkey.update(
                where={'hash': key_hash},
                data={'isActive': False}
            )
            
            if result:
                logger.info(f"Revoked API key: {result.key}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False

    async def list_keys(self) -> List[APIKey]:
        """List all API keys"""
        try:
            auth_keys = await self.db.authkey.find_many(
                order={'createdAt': 'desc'}
            )
            
            return [self._auth_key_to_api_key(auth_key) for auth_key in auth_keys]
            
        except Exception as e:
            logger.error(f"Failed to list API keys: {e}")
            return []

    def _auth_key_to_api_key(self, auth_key) -> APIKey:
        """Convert Prisma AuthKey model to APIKey pydantic model"""
        return APIKey(
            key_id=auth_key.key,
            key_hash=auth_key.hash,
            name=auth_key.name,
            key_type=APIKeyType(auth_key.keyType),
            created_at=auth_key.createdAt,
            last_used=auth_key.lastUsed,
            usage_count=auth_key.usageCount,
            is_active=auth_key.isActive,
            requests_per_minute=auth_key.requestsPerMinute,
            requests_per_hour=auth_key.requestsPerHour,
            requests_per_day=auth_key.requestsPerDay,
            permissions=auth_key.permissions,
            total_audio_minutes=auth_key.totalAudioMinutes,
            total_requests=auth_key.totalRequests,
        )


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

# Global instances - will be initialized when app starts
api_key_store: Optional[PrismaAPIKeyStore] = None
rate_limiter = RateLimiter()


def initialize_auth_store(db: Prisma):
    """Initialize the API key store with database connection"""
    global api_key_store
    api_key_store = PrismaAPIKeyStore(db)


class BearerTokenAuth(HTTPBearer):
    """OpenAI-style Bearer token authentication"""

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> APIKey:
        if api_key_store is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth service not initialized")
            
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)

        if not credentials or not credentials.credentials:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")

        api_key = credentials.credentials

        if not any(api_key.startswith(kt.value) for kt in APIKeyType):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key format")

        key_info = await api_key_store.get_key(api_key)
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
    """Dependency factory for checking permissions"""
    def check_permission(key_info: APIKey = Depends(auth)):
        if permission not in key_info.permissions:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission '{permission}' required")
        return key_info
    return check_permission


async def update_usage(request: Request, audio_duration: float = 0.0):
    """Update API key usage statistics (tracks both request and audio duration)"""
    if hasattr(request.state, 'api_key') and api_key_store:
        await api_key_store.update_key_usage(request.state.api_key, audio_duration, track_request=True)

async def update_request_usage(request: Request):
    """Track API request usage only (for when request is made)"""
    if hasattr(request.state, 'api_key') and api_key_store:
        await api_key_store.update_key_usage(request.state.api_key, audio_duration=0.0, track_request=True)

async def update_audio_usage(api_key: str, audio_duration: float):
    """Track audio duration usage only (for background processing)"""
    if api_key_store:
        await api_key_store.update_key_usage(api_key, audio_duration, track_request=False)


def get_rate_limit_headers(request: Request) -> Dict[str, str]:
    """Get rate limit headers for response"""
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
