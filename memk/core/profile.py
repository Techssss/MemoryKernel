"""Runtime performance profile selection for MemoryKernel.

Profiles keep the default local-agent experience fast and small while leaving
heavier semantic retrieval available as an explicit opt-in.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PerformanceProfile:
    name: str
    default_embedder: str
    index_mode: str
    use_fts: bool
    use_ram_index: bool
    enable_graph_index: bool
    enable_spacy: bool
    lazy_job_workers: bool
    sqlite_cache_size: int
    sqlite_mmap_size: int
    fts_candidate_limit: int

    def as_dict(self) -> dict:
        return asdict(self)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_performance_profile() -> PerformanceProfile:
    """Return the active performance profile.

    Environment:
      - MEMK_PROFILE=lite|balanced|quality
      - MEMK_INDEX_MODE=sqlite|ram
      - MEMK_GRAPH=0|1
      - MEMK_SPACY=0|1
    """
    raw_profile = os.getenv("MEMK_PROFILE", "lite").strip().lower()
    if raw_profile not in {"lite", "balanced", "quality"}:
        raw_profile = "lite"

    index_mode = os.getenv("MEMK_INDEX_MODE", "sqlite").strip().lower()
    if index_mode not in {"sqlite", "ram"}:
        index_mode = "sqlite"
    use_ram_index = index_mode == "ram"

    if raw_profile == "quality":
        default_embedder = "auto"
        cache_size = -64000
        mmap_size = 268435456
        candidate_limit = 300
        graph_default = False
        spacy_default = False
        lazy_jobs = False
    elif raw_profile == "balanced":
        default_embedder = "hashing"
        cache_size = -32000
        mmap_size = 67108864
        candidate_limit = 250
        graph_default = False
        spacy_default = False
        lazy_jobs = True
    else:
        default_embedder = "hashing"
        cache_size = -16000
        mmap_size = 33554432
        candidate_limit = 200
        graph_default = False
        spacy_default = False
        lazy_jobs = True

    return PerformanceProfile(
        name=raw_profile,
        default_embedder=default_embedder,
        index_mode=index_mode,
        use_fts=True,
        use_ram_index=use_ram_index,
        enable_graph_index=_bool_env("MEMK_GRAPH", graph_default),
        enable_spacy=_bool_env("MEMK_SPACY", spacy_default),
        lazy_job_workers=lazy_jobs,
        sqlite_cache_size=cache_size,
        sqlite_mmap_size=mmap_size,
        fts_candidate_limit=int(os.getenv("MEMK_FTS_CANDIDATES", str(candidate_limit))),
    )
