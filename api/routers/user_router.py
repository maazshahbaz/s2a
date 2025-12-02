from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from db_services.user import UserService
from security.hmac import verify_hmac
from generated.prisma.models import User

router = APIRouter(
    prefix="/users",
    tags=["Users"],
    dependencies=[Depends(verify_hmac)]
)

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    external_id: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    # Add other updatable fields here if needed

class UserResponse(BaseModel):
    id: int
    key: str
    email: str
    name: Optional[str]
    externalId: str
    createdAt: Any # Using Any to avoid datetime serialization issues for now, or use datetime
    updatedAt: Any

    class Config:
        orm_mode = True

# Dependency to get UserService
def get_user_service(request: Request) -> UserService:
    db = getattr(request.app.state, "db", None)
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return UserService(db)

@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    service: UserService = Depends(get_user_service)
):
    """Create a new user"""
    # Check if user already exists
    existing_user = await service.get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this email already exists")
    
    existing_external = await service.get_user_by_external_id(user_data.external_id)
    if existing_external:
        raise HTTPException(status_code=400, detail="User with this external ID already exists")

    try:
        user = await service.create_user(
            email=user_data.email,
            name=user_data.name,
            external_id=user_data.external_id
        )
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    service: UserService = Depends(get_user_service)
):
    """Get user by ID"""
    user = await service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.get("/by-email/{email}", response_model=UserResponse)
async def get_user_by_email(
    email: str,
    service: UserService = Depends(get_user_service)
):
    """Get user by email"""
    user = await service.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.get("/by-external-id/{external_id}", response_model=UserResponse)
async def get_user_by_external_id(
    external_id: str,
    service: UserService = Depends(get_user_service)
):
    """Get user by external ID"""
    user = await service.get_user_by_external_id(external_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    service: UserService = Depends(get_user_service)
):
    """Update user"""
    # Check if user exists
    existing_user = await service.get_user(user_id)
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        # Filter out None values
        update_data = {k: v for k, v in user_data.dict().items() if v is not None}
        if not update_data:
            return existing_user

        user = await service.update_user(user_id, update_data)
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    service: UserService = Depends(get_user_service)
):
    """Delete user"""
    # Check if user exists
    existing_user = await service.get_user(user_id)
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        await service.delete_user(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
