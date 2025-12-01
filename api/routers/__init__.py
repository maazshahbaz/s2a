from .stats import router as stats_router
from .transcribe import router as transcribe_router
from .webhook import router as webhook_router

all_routers = [stats_router, transcribe_router, webhook_router]