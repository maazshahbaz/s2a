from .stats import router as stats_router
from .transcribe import router as transcribe_router
from .user_router import router as user_router

all_routers = [stats_router, transcribe_router, user_router]