import os, time, hmac, hashlib
from fastapi import Header, HTTPException, Request
from typing import Union
from dotenv import load_dotenv

load_dotenv()

HMAC_SECRET = os.getenv("HMAC_SECRET")
VALID_WINDOW_MS = int(os.getenv("HMAC_VALID_WINDOW_MS", "300000"))  # 5 minutes

if not HMAC_SECRET:
    raise RuntimeError("HMAC_SECRET must be set")


def compute_signature(user_id: str, timestamp: str, body_hash: str):
    payload = f"{user_id}:{timestamp}:{body_hash}"
    return hmac.new(
        HMAC_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()


async def verify_hmac(
    request: Request,
    x_user_id: str = Header(..., alias="x-user-id"),
    x_timestamp: str = Header(..., alias="x-timestamp"),
    x_signature: str = Header(..., alias="x-signature"),
):
    # ---- Step 1: Validate timestamp ----
    try:
        ts = int(x_timestamp)
    except ValueError:
        raise HTTPException(401, "Invalid timestamp")

    now_ms = int(time.time() * 1000)
    if abs(now_ms - ts) > VALID_WINDOW_MS:
        raise HTTPException(401, "Request expired")

    # ---- Step 2: Read request body (async!) ----
    body = await request.body()
    body_hash = hashlib.sha256(body).hexdigest()

    # ---- Step 3: Recompute expected signature ----
    expected = compute_signature(x_user_id, x_timestamp, body_hash)
    
    # DEBUG LOGS
    print(f"[HMAC DEBUG] Secret (first 4): {HMAC_SECRET[:4] if HMAC_SECRET else 'None'}")
    print(f"[HMAC DEBUG] User ID: {x_user_id}")
    print(f"[HMAC DEBUG] Timestamp: {x_timestamp}")
    print(f"[HMAC DEBUG] Body Hash: {body_hash}")
    print(f"[HMAC DEBUG] Expected Sig: {expected}")
    print(f"[HMAC DEBUG] Received Sig: {x_signature}")

    if not hmac.compare_digest(expected, x_signature):
        raise HTTPException(401, "Invalid HMAC signature")

    # ---- Step 4: Return validated user_id ----
    try:
        return int(x_user_id)
    except ValueError:
        # Allow string IDs (e.g. external IDs during user creation)
        return x_user_id
