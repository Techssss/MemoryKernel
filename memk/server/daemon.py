import logging
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from memk.core.service import MemoryKernelService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memk.daemon")

app = FastAPI(title="MemoryKernel Daemon", version="0.1.0")
service = MemoryKernelService()

class AddRequest(BaseModel):
    content: str
    importance: float = 0.5
    confidence: float = 1.0
    workspace_id: str = "default"

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    workspace_id: str = "default"

class ContextRequest(BaseModel):
    query: str
    max_chars: int = 500
    threshold: float = 0.3
    workspace_id: str = "default"

@app.on_event("startup")
def startup_event():
    service.ensure_initialized()
    logger.info("Daemon ready (HTTP Transport Adapter active).")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "diagnostics": service.get_diagnostics()
    }

@app.post("/add")
async def add_memory(req: AddRequest):
    try:
        return await service.add_memory(req.content, req.importance, req.confidence, req.workspace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(req: SearchRequest):
    try:
        results = await service.search(req.query, req.limit, req.workspace_id)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context")
async def build_context(req: ContextRequest):
    try:
        context_str = await service.build_context(req.query, req.max_chars, req.threshold, req.workspace_id)
        return {"context": context_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/doctor")
def doctor():
    try:
        return service.get_diagnostics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Jobs ---

@app.get("/jobs")
def list_jobs():
    return {"jobs": service.runtime.jobs.list_jobs()}

@app.post("/jobs/synthesize")
def submit_synthesis():
    job_id = service.submit_job("synthesize", "synthesize_all")
    return {"job_id": job_id, "status": "pending"}

@app.post("/shutdown")
def shutdown():
    import os, signal
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting down"}
