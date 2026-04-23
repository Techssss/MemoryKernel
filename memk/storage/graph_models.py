"""
memk.storage.graph_models
=========================
Data models for the knowledge graph sidecar tables (V5 schema).

These are plain dataclasses — no ORM, no Pydantic dependency here.
Keeps the storage layer lightweight and consistent with the rest of memk
(IndexEntry in retrieval/index.py uses the same pattern).
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------

_MULTI_SPACE = re.compile(r"\s+")


def normalize_entity_text(text: str) -> str:
    """
    Canonical normalization for entity dedup.

    Rules
    -----
    1. Strip leading/trailing whitespace
    2. Lowercase
    3. Collapse consecutive whitespace to single space

    Examples
    --------
    >>> normalize_entity_text("  Google   LLC ")
    'google llc'
    >>> normalize_entity_text("fastAPI")
    'fastapi'
    """
    return _MULTI_SPACE.sub(" ", text.strip().lower())


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class EntityRecord:
    """
    A unique real-world entity within a workspace.

    id              : Auto-incremented integer PK (assigned by SQLite).
    workspace_id    : Workspace this entity belongs to.
    canonical_text  : Human-readable display form ("Google LLC").
    normalized_text : Lowercase/stripped dedup key ("google llc").
    entity_type     : Optional type tag ("ORG", "PERSON", "TECH", etc.).
    first_seen_ts   : ISO 8601 timestamp of first observation.
    last_seen_ts    : ISO 8601 timestamp of most recent observation.
    confidence      : Aggregated confidence [0, 1].
    """
    id: int
    workspace_id: str
    canonical_text: str
    normalized_text: str
    entity_type: Optional[str] = None
    first_seen_ts: str = ""
    last_seen_ts: str = ""
    confidence: float = 0.5


@dataclass
class MentionRecord:
    """
    Links a memory row to an entity it references.

    Composite PK: (memory_id, entity_id, role_hint).
    Stored WITHOUT ROWID for compact B-tree storage.

    memory_id   : FK to memories.id.
    entity_id   : FK to entity.id.
    start_char  : Optional character offset of the entity span in memory text.
    end_char    : Optional end offset.
    role_hint   : "subject" | "object" | "context" | None.
    weight      : Importance weight of this mention [0, 1].
    """
    memory_id: str
    entity_id: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    role_hint: Optional[str] = None
    weight: float = 1.0


@dataclass
class EdgeRecord:
    """
    A directional relationship between two entities.

    Extracted from a specific memory (provenance_memory_id),
    so we always know *why* this edge exists.

    id                   : Auto-incremented integer PK.
    workspace_id         : Workspace scope.
    src_entity_id        : FK to entity.id (source).
    rel_type             : Relationship label ("manages", "uses", "located_in").
    dst_entity_id        : FK to entity.id (destination).
    weight               : Edge strength [0, 1].
    confidence           : Extraction confidence [0, 1].
    provenance_memory_id : FK to memories.id — the memory this was extracted from.
    archived             : Soft-delete flag (0=active, 1=archived).
    created_at           : ISO 8601 timestamp.
    """
    id: int
    workspace_id: str
    src_entity_id: int
    rel_type: str
    dst_entity_id: int
    weight: float = 1.0
    confidence: float = 0.5
    provenance_memory_id: str = ""
    archived: int = 0
    created_at: str = ""


@dataclass
class KGFactRecord:
    """
    Consolidated knowledge — a summary produced by the consolidation pipeline.

    Separate from the existing `facts` table to avoid coupling with
    the legacy SPO triplet storage.

    id              : UUID string PK.
    workspace_id    : Workspace scope.
    canonical_text  : Human-readable summary text.
    summary_json    : Optional JSON blob with structured details.
    confidence      : Aggregated confidence [0, 1].
    created_ts      : ISO 8601 timestamp.
    """
    id: str
    workspace_id: str
    canonical_text: str
    summary_json: Optional[str] = None
    confidence: float = 0.5
    created_ts: str = ""
