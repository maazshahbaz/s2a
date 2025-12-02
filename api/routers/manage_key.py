from fastapi import APIRouter, Request, Depends, HTTPException, status, Form
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from security.hmac import verify_hmac
from db_services.auth import PrismaAPIKeyStore, APIKeyType
from db_services.user import UserService
from generated.prisma.models import User
from dependencies import get_user_service, get_auth_service, get_db

router = APIRouter(prefix="/api-keys", tags=["API Keys"])

# --- Models ---

class CreateKeyRequest(BaseModel):
    name: str

class CreateKeyResponse(BaseModel):
    api_key: str
    name: str
    key_type: str

class ApiKeyResponse(BaseModel):
    id: str # This is the key_id (e.g. bp-proj-...)
    name: str
    masked_key: str
    created_at: Optional[str]
    last_used: Optional[str]
    is_active: bool
    usage_count: int


async def get_current_user_id(
    external_id: str = Depends(verify_hmac),
    user_service: UserService = Depends(get_user_service)
) -> int:
    """
    Resolves the external_id (from HMAC) to the internal database user ID.
    """
    user = await user_service.get_user_by_external_id(external_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.id

# --- Endpoints ---

@router.post("/", response_model=CreateKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_key(
    key_data: CreateKeyRequest,
    user_id: int = Depends(get_current_user_id),
    auth_service: PrismaAPIKeyStore = Depends(get_auth_service)
):
    """Create a new API key for the authenticated user"""
    try:
        # Create key using the store
        api_key, key_info = await auth_service.create_key(
            user_id=user_id,
            name=key_data.name,
            key_type=APIKeyType.PROJECT
        )
        
        return CreateKeyResponse(
            api_key=api_key,
            name=key_info.name,
            key_type=key_info.key_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[ApiKeyResponse])
async def list_keys(
    user_id: int = Depends(get_current_user_id),
    auth_service: PrismaAPIKeyStore = Depends(get_auth_service)
):
    """List all API keys for the authenticated user"""
    try:
        keys = await auth_service.list_keys(user_id)
        
        response_keys = []
        for k in keys:
            response_keys.append(ApiKeyResponse(
                id=k.key_id,
                name=k.name,
                masked_key=k.masked_key if k.masked_key else f"{k.key_type.value}-{'*' * 24}", # Use stored mask or fallback
                created_at=k.created_at.isoformat() if k.created_at else None,
                last_used=k.last_used.isoformat() if k.last_used else None,
                is_active=k.is_active,
                usage_count=k.usage_count
            ))
            
        return response_keys
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{key_id}/revoke", status_code=status.HTTP_200_OK)
async def revoke_key_endpoint(
    key_id: str,
    user_id: int = Depends(get_current_user_id),
    db = Depends(get_db)
):
    """Revoke an API key"""
    try:
        # Verify ownership using the UUID (key)
        # In Prisma schema: key String @unique @default(uuid())
        # So key_id passed here is the 'key' column in DB.
        
        auth_key = await db.authkey.find_first(
            where={
                'key': key_id,
                'userId': user_id
            }
        )
        
        if not auth_key:
            raise HTTPException(status_code=404, detail="Key not found")
            
        # Revoke
        await db.authkey.update(
            where={'id': auth_key.id}, # Use internal int ID for update
            data={'isActive': False}
        )
        
        return {"status": "revoked", "id": key_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
