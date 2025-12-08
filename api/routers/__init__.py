from .stats import router as stats_router
from .transcribe import router as transcribe_router
from .webhook import router as webhook_router
from .user_router import router as user_router
from .manage_key import router as manage_key_router

all_routers = [stats_router, transcribe_router, user_router, manage_key_router, webhook_router]
