from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional
import uuid

class WorkspaceManifest(BaseModel):
    brain_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workspace_root: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "1.0"
    generation: int = 0
    embedding_model_id: str = "default"
    index_version: int = 1

class ResponseMetadata(BaseModel):
    """Metadata attached to every service response for consistency tracking."""
    workspace_id: str
    generation: int
    cache_hit: bool = False
    degraded: bool = False
    stale_warning: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
