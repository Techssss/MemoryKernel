"""
memk.api.v1
===========
Version 1 of the public MemoryKernel API.

Stable contract for external integrations.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from memk.core.service import MemoryKernelService
from memk.api.models import (
    RememberRequest, SearchRequest, ContextRequest, IngestGitRequest,
    APIResponse, APIMetadata,
    RememberResponse, SearchResponse, ContextResponse, StatusResponse, IngestGitResponse,
    MemoryItem
)

logger = logging.getLogger("memk.api.v1")

router = APIRouter(prefix="/v1", tags=["v1"])

# Shared service instance
_service: Optional[MemoryKernelService] = None


def get_service() -> MemoryKernelService:
    """Get or create service instance."""
    global _service
    if _service is None:
        _service = MemoryKernelService()
    return _service


def resolve_workspace_id(workspace_id: Optional[str]) -> str:
    """Resolve workspace ID from request or auto-detect."""
    if workspace_id:
        return workspace_id
    
    # Auto-detect from current workspace
    try:
        from memk.workspace.manager import WorkspaceManager
        ws = WorkspaceManager()
        if ws.is_initialized():
            return ws.get_manifest().brain_id
    except Exception:
        pass
    
    return "default"


# ---------------------------------------------------------------------------
# Core Operations
# ---------------------------------------------------------------------------

@router.post("/remember", response_model=APIResponse)
async def remember(req: RememberRequest):
    """
    Add a memory to the workspace.
    
    This is the primary write operation. Memories are embedded and indexed
    for later retrieval.
    """
    try:
        service = get_service()
        workspace_id = resolve_workspace_id(req.workspace_id)
        
        result = await service.add_memory(
            req.content,
            req.importance,
            req.confidence,
            workspace_id
        )
        
        # Extract data and metadata
        data = {
            "id": result["id"],
            "extracted_facts": result.get("extracted_facts", [])
        }
        
        metadata = APIMetadata(**result["metadata"])
        
        return APIResponse(data=data, metadata=metadata)
        
    except Exception as e:
        logger.error(f"Remember failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=APIResponse)
async def search(req: SearchRequest):
    """
    Search for relevant memories.
    
    Returns memories ranked by relevance to the query.
    Supports staleness detection via client_generation.
    """
    try:
        service = get_service()
        workspace_id = resolve_workspace_id(req.workspace_id)
        
        result = await service.search(
            req.query,
            req.limit,
            workspace_id,
            req.client_generation
        )
        
        # Convert results to MemoryItem models
        items = [MemoryItem(**item) for item in result["results"]]
        
        data = {"results": [item.model_dump() for item in items]}
        metadata = APIMetadata(**result["metadata"])
        
        return APIResponse(data=data, metadata=metadata)
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context", response_model=APIResponse)
async def context(req: ContextRequest):
    """
    Build RAG context from relevant memories.
    
    Returns a formatted context string suitable for LLM prompts.
    Respects max_chars limit and relevance threshold.
    """
    try:
        service = get_service()
        workspace_id = resolve_workspace_id(req.workspace_id)
        
        result = await service.build_context(
            req.query,
            req.max_chars,
            req.threshold,
            workspace_id,
            req.client_generation
        )
        
        data = {
            "context": result["context"],
            "char_count": len(result["context"])
        }
        
        metadata = APIMetadata(**result["metadata"])
        
        return APIResponse(data=data, metadata=metadata)
        
    except Exception as e:
        logger.error(f"Context build failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=APIResponse)
async def status(workspace_id: Optional[str] = Query(None)):
    """
    Get workspace status and statistics.
    
    Returns current generation, memory counts, and watcher status.
    """
    try:
        from memk.workspace.manager import WorkspaceManager
        
        workspace_id = resolve_workspace_id(workspace_id)
        service = get_service()
        
        # Get workspace info
        ws = WorkspaceManager()
        ws_info = ws.get_status_info()
        
        # Get diagnostics
        diag = service.get_diagnostics(workspace_id)
        
        # Get watcher status
        watcher_status = None
        try:
            from memk.server.daemon import _watchers
            watcher = _watchers.get(workspace_id)
            if watcher:
                watcher_status = watcher.get_status()
        except:
            pass
        
        data = {
            "workspace_id": workspace_id,
            "generation": ws_info.get("generation", 0),
            "initialized": ws_info.get("initialized", False),
            "workspace_root": ws_info.get("root", ""),
            "stats": diag.get("db_stats", {}),
            "watcher": watcher_status
        }
        
        metadata = APIMetadata(
            workspace_id=workspace_id,
            generation=ws_info.get("generation", 0)
        )
        
        return APIResponse(data=data, metadata=metadata)
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/git", response_model=APIResponse)
async def ingest_git(req: IngestGitRequest):
    """
    Ingest knowledge from Git commit history.
    
    Analyzes recent commits and extracts meaningful memories.
    """
    try:
        from memk.ingestion.git_ingestor import GitIngestor
        from memk.workspace.manager import WorkspaceManager
        
        workspace_id = resolve_workspace_id(req.workspace_id)
        service = get_service()
        
        # Get repo path
        ws = WorkspaceManager()
        repo_path = ws.root
        
        # Ingest commits
        ingestor = GitIngestor(repo_path=str(repo_path))
        memories = ingestor.ingest_commits(
            limit=req.limit,
            since=req.since,
            branch=req.branch
        )
        
        # Add memories to brain
        added_count = 0
        categories = {}
        
        for mem in memories:
            try:
                await service.add_memory(
                    mem["content"],
                    importance=mem["importance"],
                    workspace_id=workspace_id
                )
                added_count += 1
                
                # Track categories
                cat = mem["metadata"]["category"]
                categories[cat] = categories.get(cat, 0) + 1
            except Exception as e:
                logger.warning(f"Failed to add memory: {e}")
        
        data = {
            "ingested_count": added_count,
            "categories": categories
        }
        
        # Get current generation
        ws_info = ws.get_status_info()
        metadata = APIMetadata(
            workspace_id=workspace_id,
            generation=ws_info.get("generation", 0)
        )
        
        return APIResponse(data=data, metadata=metadata)
        
    except Exception as e:
        logger.error(f"Git ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """
    Health check endpoint.
    
    Returns OK if service is running.
    """
    return {"status": "ok", "version": "1.0.0"}


@router.get("/metrics", response_model=APIResponse)
async def metrics(workspace_id: Optional[str] = Query(None)):
    """
    Get production metrics and observability data.
    
    Returns request latency, cache performance, database stats, and job status.
    """
    try:
        from memk.core.metrics import get_metrics_collector
        from memk.core.tracing import get_collector as get_trace_collector
        from memk.storage.config import get_wal_status
        from memk.workspace.manager import WorkspaceManager
        
        workspace_id = resolve_workspace_id(workspace_id)
        service = get_service()
        
        # Get metrics
        metrics_collector = get_metrics_collector()
        metrics_summary = metrics_collector.get_metrics_summary()
        
        # Get trace report
        trace_collector = get_trace_collector()
        trace_report = trace_collector.get_report()
        
        # Get diagnostics
        diag = service.get_diagnostics(workspace_id)
        db_stats = diag.get("db_stats", {})
        
        # Get WAL status
        ws = WorkspaceManager()
        if ws.is_initialized():
            db_path = ws.get_db_path()
            wal_status = get_wal_status(db_path)
        else:
            wal_status = {}
        
        # Get job status
        runtime = diag.get("runtime", {})
        
        data = {
            "requests": metrics_summary.get("requests", {}),
            "latency": metrics_summary.get("latency", {}),
            "cache": metrics_summary.get("cache", {}),
            "degraded": metrics_summary.get("degraded", {}),
            "operations": metrics_summary.get("operations", {}),
            "database": {
                "size_mb": db_stats.get("database_size_mb", 0),
                "wal_size_mb": wal_status.get("wal_size_mb", 0),
                "total_memories": db_stats.get("total_memories", 0),
                "total_facts": db_stats.get("total_active_facts", 0),
                "schema_version": db_stats.get("schema_version", 0),
            },
            "jobs": {
                "active": runtime.get("active_jobs", 0),
                "queue_depth": 0,  # Would need to expose from job manager
            },
            "traces": {
                "total_requests": trace_report.get("total_requests", 0),
                "slow_request_count": trace_report.get("slow_request_count", 0),
                "slow_rate_pct": trace_report.get("slow_rate_pct", 0),
                "latency_percentiles": trace_report.get("latency_percentiles", {}),
            },
            "uptime_seconds": metrics_summary.get("uptime_seconds", 0),
        }
        
        # Get current generation
        ws_info = ws.get_status_info()
        metadata = APIMetadata(
            workspace_id=workspace_id,
            generation=ws_info.get("generation", 0)
        )
        
        return APIResponse(data=data, metadata=metadata)
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

