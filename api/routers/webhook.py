from fastapi import APIRouter, Request, HTTPException
from datetime import datetime
from pathlib import Path
import json
import os

# Configuration
WEBHOOK_FILE = "/app/logs/webhook_data.jsonl"  # JSON Lines format

router = APIRouter(prefix="/webhook", tags=["Webhook"])

@router.post("")
async def webhook_handler(request: Request):
    """
    Webhook endpoint that appends incoming data to a file.
    Creates the file if it doesn't exist.
    """
    try:
        # Get the request body
        body = await request.json()
        
        # Create a log entry with timestamp
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "data": body,
            "headers": dict(request.headers),
            "method": request.method,
            "url": str(request.url)
        }
        
        # Ensure file exists and append to it
        webhook_path = Path(WEBHOOK_FILE)
        
        # Create parent directory if it doesn't exist
        webhook_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to file (creates if doesn't exist)
        with open(webhook_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            f.flush()  # Force write to disk
        
        return {
            "status": "success",
            "message": "Webhook data saved",
            "timestamp": log_entry["timestamp"],
            "file_path": str(webhook_path.absolute())
        }
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_logs(limit: int = 100):
    """
    Retrieve the most recent webhook logs.
    """
    if not os.path.exists(WEBHOOK_FILE):
        return {"logs": [], "message": "No logs found"}
    
    try:
        logs = []
        with open(WEBHOOK_FILE, "r") as f:
            lines = f.readlines()
            # Get last N lines
            for line in lines[-limit:]:
                if line.strip():  # Skip empty lines
                    logs.append(json.loads(line.strip()))
        
        return {
            "logs": logs,
            "count": len(logs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/logs")
async def clear_logs():
    """
    Clear all webhook logs.
    """
    try:
        if os.path.exists(WEBHOOK_FILE):
            os.remove(WEBHOOK_FILE)
            return {"status": "success", "message": "Logs cleared"}
        return {"status": "success", "message": "No logs to clear"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))