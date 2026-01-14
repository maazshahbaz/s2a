from typing import Optional, Dict, Any, List
from generated.prisma import Prisma
from generated.prisma.models import User
import math
from loguru import logger


def sanitize_json_data(data: Any) -> Any:
    """Sanitize data to be JSON-compatible by handling NaN, inf, and numpy types"""
    if data is None:
        return None
    elif isinstance(data, dict):
        return {key: sanitize_json_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_json_data(item) for item in data]
    elif isinstance(data, (float, complex)):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)
    elif hasattr(data, 'item'):  # numpy types
        return sanitize_json_data(data.item())
    elif hasattr(data, 'tolist'):  # numpy arrays
        return sanitize_json_data(data.tolist())
    else:
        return data


class UserService:
    def __init__(self, db: Prisma):
        self.db = db

    async def create_user(
        self,
        email: str,
        name: Optional[str],
        external_id: str
    ) -> User:
        """Create a new user"""
        user = await self.db.user.create(
            data={
                'email': email,
                'name': name,
                'externalId': external_id
            }
        )
        logger.info(f"Created user {email} {name}")
        return user

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by id"""
        return await self.db.user.find_unique(
            where={'id': user_id},
        )

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return await self.db.user.find_unique(
            where={'email': email},
        )

    async def get_user_by_external_id(self, external_id: str) -> Optional[User]:
        """Get user by external_id"""
        return await self.db.user.find_unique(
            where={'externalId': external_id},
        )

    async def update_user(
       self,
       user_id: int,
       data: Dict[str, Any]
    ) -> Optional[User]:
        """Update user"""
        # Sanitize data before update
        sanitized_data = sanitize_json_data(data)
        
        user = await self.db.user.update(
            where={'id': user_id},
            data=sanitized_data
        )
        logger.info(f"Updated user {user_id}")
        return user

    async def delete_user(self, user_id: int) -> Optional[User]:
        """Delete user by id"""
        user = await self.db.user.delete(
            where={'id': user_id}
        )
        logger.info(f"Deleted user {user_id}")
        return user