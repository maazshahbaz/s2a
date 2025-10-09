from .stats import router as stats_router
from .transcribe import router as transcribe_router
from .intelligence import router as intelligence_router

all_routers = [stats_router, transcribe_router, intelligence_router]