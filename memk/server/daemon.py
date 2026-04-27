import logging
import time
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from memk.core.service import MemoryKernelService
from memk.core.runtime import get_runtime
from memk.api.v1 import router as v1_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memk.daemon")

app = FastAPI(title="MemoryKernel Daemon", version="1.0.0")

# Include v1 API router
app.include_router(v1_router)
import os
os.environ["MEMK_DAEMON_MODE"] = "1"
service = MemoryKernelService()

# Global watcher registry (workspace_id -> WatcherService)
_watchers: Dict[str, Any] = {}

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
        "version": "1.0.0",
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
        return await service.search(req.query, req.limit, req.workspace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context")
async def build_context(req: ContextRequest):
    try:
        return await service.build_context(req.query, req.max_chars, req.threshold, req.workspace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/doctor")
def doctor(workspace_id: str = "default"):
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
