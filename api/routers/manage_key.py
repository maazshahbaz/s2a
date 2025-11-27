from fastapi import APIRouter, Request, Depends, HTTPException
from pydantic import BaseModel
from db import init_db, engine, get_session
from sqlmodel import Session
from security.hmac import verify_hmac
from dependencies import get_auth_service
import os, secrets

router = APIRouter(prefix="/keys")

class CreateKeyOut(BaseModel):
    api_key: str
    name: str

@router.post("/create", response_model=CreateKeyOut)
async def create_key(req: Request, name:str = Form(...),user_id: int = Depends(verify_hmac), auth_svc = Depends(get_auth_service)):
    # verify_hmac returns the Next.js-signed user id; but ensure admin privileges if needed
   
    ak = auth_svc.create_key(user_id, name,'bp-svc')
    return CreateKeyOut(api_key=ak.api_key, name=ak.key_info.name)

@router.get("/list")
async def list_keys(req: Request, user_id: int = Depends(verify_hmac), auth_svc = Depends(get_auth_service)):
    try:
        keys = auth_svc.list_keys(user_id)
        return {"keys": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/revoke")
async def revoke_key(req: Request, key_id: str, user_id: int = Depends(verify_hmac), auth_svc = Depends(get_auth_service)):
    try:
        deleted_key = auth_svc.revoke_key(key_id)
        return {"status":"revoked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
