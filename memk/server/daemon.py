import logging
import time
import asyncio
import secrets
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from memk.core.service import MemoryKernelService
from memk.core.runtime import get_runtime
from memk.core.metrics import record_request
from memk.api.v1 import router as v1_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memk.daemon")

app = FastAPI(title="MemoryKernel Daemon", version="0.1.0")

# Include v1 API router
app.include_router(v1_router)
import os
os.environ["MEMK_DAEMON_MODE"] = "1"
service = MemoryKernelService()

# Global watcher registry (workspace_id -> WatcherService)
_watchers: Dict[str, Any] = {}

PUBLIC_PATHS = {"/health", "/v1/health", "/docs", "/redoc", "/openapi.json"}


def _configured_api_token() -> str:
    return os.getenv("MEMK_API_TOKEN", "").strip()


def _request_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-memk-token", "").strip()


def _is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith("/docs/")


def _mark_deprecated(response: Response, successor: str) -> None:
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = f"<{successor}>; rel=\"successor-version\""

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
async def startup_event():
    os.environ["MEMK_DAEMON_MODE"] = "1"
    # Shared global models are initialized once
    service.global_runtime.initialize_global()
    logger.info("Daemon ready (Global Infrastructure active).")
    # Start eviction task
    asyncio.create_task(eviction_background_task())
    # Auto-start watcher for default workspace if initialized
    asyncio.create_task(auto_start_watchers())

async def auto_start_watchers():
    """Auto-start file watchers for initialized workspaces."""
    await asyncio.sleep(2)  # Wait for daemon to fully start
    try:
        from memk.workspace.manager import WorkspaceManager
        from memk.watcher.file_watcher import WatcherService, WATCHDOG_AVAILABLE
        
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available - file watching disabled")
            return
        
        # Try to start watcher for current workspace
        try:
            ws = WorkspaceManager()
            if ws.is_initialized():
                manifest = ws.get_manifest()
                workspace_id = manifest.brain_id
                
                logger.info(f"Auto-starting file watcher for workspace: {workspace_id}")
                watcher = WatcherService(str(ws.root), ws)
                watcher.start()
                _watchers[workspace_id] = watcher
                logger.info(f"File watcher started for workspace: {workspace_id}")
        except Exception as e:
            logger.warning(f"Could not auto-start watcher: {e}")
    except Exception as e:
        logger.error(f"Error in auto_start_watchers: {e}")

async def eviction_background_task():
    while True:
        await asyncio.sleep(60) # Check every minute
        try:
            get_runtime().evict_idle_workspaces(idle_seconds=1800) # 30 mins
        except Exception as e:
            logger.error(f"Error in eviction task: {e}")

@app.middleware("http")
async def request_observability_and_auth(request: Request, call_next):
    start_time = time.perf_counter()
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = _configured_api_token()
    path = request.url.path

    if token and not _is_public_path(path):
        provided = _request_token(request)
        if not provided or not secrets.compare_digest(provided, token):
            process_time = time.perf_counter() - start_time
            record_request(
                f"{request.method} {path}",
                process_time * 1000,
                degraded=True,
                status_code=401,
                error=True,
            )
            response = JSONResponse(
                status_code=401,
                content={
                    "detail": {
                        "code": "auth_required",
                        "message": "Missing or invalid MemoryKernel API token.",
                    }
                },
            )
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            logger.warning(
                "request rejected method=%s path=%s status=401 request_id=%s",
                request.method,
                path,
                request_id,
            )
            return response

    try:
        response = await call_next(request)
    except Exception:
        process_time = time.perf_counter() - start_time
        record_request(
            f"{request.method} {path}",
            process_time * 1000,
            degraded=True,
            status_code=500,
            error=True,
        )
        logger.exception(
            "request failed method=%s path=%s request_id=%s",
            request.method,
            path,
            request_id,
        )
        raise

    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    record_request(
        f"{request.method} {path}",
        process_time * 1000,
        degraded=response.status_code >= 500,
        status_code=response.status_code,
        error=response.status_code >= 400,
    )
    logger.info(
        "request method=%s path=%s status=%s duration_ms=%.2f request_id=%s",
        request.method,
        path,
        response.status_code,
        process_time * 1000,
        request_id,
    )
    return response

@app.get("/health")
def health(response: Response):
    _mark_deprecated(response, "/v1/health")
    return {
        "status": "ok", 
        "version": "0.1.0",
        "auth_enabled": bool(_configured_api_token()),
        "diagnostics": service.get_diagnostics()
    }

@app.post("/add")
async def add_memory(req: AddRequest, response: Response):
    _mark_deprecated(response, "/v1/remember")
    try:
        return await service.add_memory(req.content, req.importance, req.confidence, req.workspace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(req: SearchRequest, response: Response):
    _mark_deprecated(response, "/v1/search")
    try:
        return await service.search(req.query, req.limit, req.workspace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context")
async def build_context(req: ContextRequest, response: Response):
    _mark_deprecated(response, "/v1/context")
    try:
        return await service.build_context(req.query, req.max_chars, req.threshold, req.workspace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/doctor")
def doctor(response: Response, workspace_id: str = "default"):
    _mark_deprecated(response, "/v1/status")
    try:
        return service.get_diagnostics(workspace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Jobs ---

@app.get("/jobs")
def list_jobs(workspace_id: str = "default"):
    runtime = service._get_runtime(workspace_id)
    return {"jobs": runtime.jobs.list_jobs()}

@app.post("/jobs/synthesize")
def submit_synthesis(workspace_id: str = "default"):
    job_id = service.submit_job(workspace_id, "synthesize")
    return {"job_id": job_id, "status": "pending"}

@app.post("/shutdown")
def shutdown():
    import os, signal
    # Stop all watchers
    for watcher in _watchers.values():
        try:
            watcher.stop()
        except:
            pass
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting down"}

# --- Watcher Endpoints ---

@app.get("/watcher/status")
def watcher_status(workspace_id: str = "default"):
    """Get file watcher status for a workspace."""
    try:
        from memk.workspace.manager import WorkspaceManager
        
        # Resolve workspace_id if default
        if workspace_id == "default":
            ws = WorkspaceManager()
            if ws.is_initialized():
                workspace_id = ws.get_manifest().brain_id
        
        watcher = _watchers.get(workspace_id)
        if not watcher:
            return {"status": {"running": False, "workspace_id": workspace_id}}
        
        return {"status": watcher.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/watcher/start")
def watcher_start(workspace_id: str = "default"):
    """Start file watcher for a workspace."""
    try:
        from memk.workspace.manager import WorkspaceManager
        from memk.watcher.file_watcher import WatcherService, WATCHDOG_AVAILABLE
        
        if not WATCHDOG_AVAILABLE:
            raise HTTPException(status_code=400, detail="watchdog library not installed")
        
        # Resolve workspace_id if default
        if workspace_id == "default":
            ws = WorkspaceManager()
            if ws.is_initialized():
                workspace_id = ws.get_manifest().brain_id
            else:
                raise HTTPException(status_code=400, detail="Workspace not initialized")
        else:
            ws = WorkspaceManager()
        
        # Check if already running
        if workspace_id in _watchers and _watchers[workspace_id].is_running():
            return {"success": False, "message": "Watcher already running"}
        
        # Start watcher
        watcher = WatcherService(str(ws.root), ws)
        watcher.start()
        _watchers[workspace_id] = watcher
        
        return {"success": True, "message": "Watcher started", "workspace_id": workspace_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/watcher/stop")
def watcher_stop(workspace_id: str = "default"):
    """Stop file watcher for a workspace."""
    try:
        from memk.workspace.manager import WorkspaceManager
        
        # Resolve workspace_id if default
        if workspace_id == "default":
            ws = WorkspaceManager()
            if ws.is_initialized():
                workspace_id = ws.get_manifest().brain_id
        
        watcher = _watchers.get(workspace_id)
        if not watcher:
            return {"success": False, "message": "No watcher running"}
        
        watcher.stop()
        del _watchers[workspace_id]
        
        return {"success": True, "message": "Watcher stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
