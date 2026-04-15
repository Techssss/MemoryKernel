"""
memk.api.models
===============
Public API request/response models for v1.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ---------------------------------------------------------------------------
# Response Envelope
# ---------------------------------------------------------------------------

class APIMetadata(BaseModel):
    """Standard metadata included in all API responses."""
    workspace_id: str
    generation: int
    cache_hit: bool = False
    degraded: bool = False
    stale_warning: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class APIResponse(BaseModel):
    """Standard response envelope for all v1 endpoints."""
    data: Dict[str, Any]
    metadata: APIMetadata


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class RememberRequest(BaseModel):
    """Request to add a memory."""
    content: str = Field(..., description="Memory content to store")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Priority/importance (0-1)")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence level (0-1)")
    workspace_id: Optional[str] = Field(None, description="Workspace ID (auto-detected if not provided)")


class SearchRequest(BaseModel):
    """Request to search memories."""
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Max results to return")
    workspace_id: Optional[str] = Field(None, description="Workspace ID")
    client_generation: Optional[int] = Field(None, description="Client's last known generation for staleness detection")


class ContextRequest(BaseModel):
    """Request to build RAG context."""
    query: str = Field(..., description="Context query")
    max_chars: int = Field(500, ge=100, le=5000, description="Maximum context length")
    threshold: float = Field(0.3, ge=0.0, le=1.0, description="Relevance threshold")
    workspace_id: Optional[str] = Field(None, description="Workspace ID")
    client_generation: Optional[int] = Field(None, description="Client's last known generation")


class IngestGitRequest(BaseModel):
    """Request to ingest Git history."""
    limit: int = Field(50, ge=1, le=1000, description="Number of commits to ingest")
    since: Optional[str] = Field(None, description="Only commits after this date (YYYY-MM-DD)")
    branch: str = Field("HEAD", description="Git branch to ingest from")
    workspace_id: Optional[str] = Field(None, description="Workspace ID")


# ---------------------------------------------------------------------------
# Response Data Models
# ---------------------------------------------------------------------------

class MemoryItem(BaseModel):
    """A single memory or fact item."""
    item_type: str
    id: str
    content: str
    score: float
    importance: float
    confidence: float
    created_at: str
    access_count: int = 0
    decay_score: float = 1.0


class RememberResponse(BaseModel):
    """Response from remember operation."""
    id: str
    extracted_facts: List[Dict[str, str]] = []


class SearchResponse(BaseModel):
    """Response from search operation."""
    results: List[MemoryItem]


class ContextResponse(BaseModel):
    """Response from context operation."""
    context: str
    char_count: int


class StatusResponse(BaseModel):
    """Response from status operation."""
    workspace_id: str
    generation: int
    initialized: bool
    workspace_root: str
    stats: Dict[str, Any]
    watcher: Optional[Dict[str, Any]] = None


class IngestGitResponse(BaseModel):
    """Response from Git ingestion."""
    ingested_count: int
    categories: Dict[str, int]

