import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from summary_client import AsyncAgentSummary, AgentSummaryResponse


# --- Request Model ---

class AgentSummaryRequest(BaseModel):
    summaries: Dict[str, str] = Field(
        ...,
        description="Dict of call summaries keyed by summary id",
        examples=[{
            "summary_1": "Customer called about pricing for 500 routers.",
            "summary_2": "Customer reported a missing charging cable.",
        }],
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional tracking ID. Auto-generated if not provided.",
    )


# --- App Lifecycle ---

summarizer: AsyncAgentSummary | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global summarizer
    summarizer = AsyncAgentSummary()
    await summarizer.initialize()
    print("Triton client initialized")
    yield
    print("Shutting down")


# --- FastAPI App ---

app = FastAPI(
    title="Agent Summary Service",
    description="Generates key update points from multiple call summaries for manager review",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post(
    "/agent-summary",
    response_model=AgentSummaryResponse,
    summary="Generate agent key updates",
    description="Accepts a dict of call summaries and returns 4-5 key update points.",
)
async def generate_summary(request: AgentSummaryRequest) -> AgentSummaryResponse:
    if not request.summaries:
        raise HTTPException(status_code=400, detail="summaries dict cannot be empty")

    request_id = request.request_id or str(uuid.uuid4())

    result = await summarizer.generate_agent_summary(
        summaries=request.summaries,
        request_id=request_id,
    )

    return result


@app.get("/health")
async def health():
    return {"status": "ok"}