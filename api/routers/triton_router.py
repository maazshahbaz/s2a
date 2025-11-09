# api/routers/triton_router.py
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["Triton Inference"])

class TritonRequest(BaseModel):
    transcription: str
    max_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9

@router.post("/analyze_call")
async def analyze_call(request: Request, body: TritonRequest):
    triton_client = getattr(request.app.state, "triton_client", None)

    if not triton_client:
        raise HTTPException(status_code=503, detail="Triton client not initialized")

    try:
        result = triton_client.analyze_call(
            transcription=body.transcription,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
