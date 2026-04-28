import asyncio
from pathlib import Path
from uuid import uuid4

from memk.core.profile import get_performance_profile
from memk.retrieval.retriever import CandidateFirstRetriever
from memk.storage.db import MemoryDB


def _db_path(name: str) -> str:
    base = Path("tmp_pytest") / "performance_profile"
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"{name}_{uuid4().hex}.db")


def test_lite_profile_defaults_to_hashing_embedder(monkeypatch):
    import memk.core.embedder as embedder_mod

    monkeypatch.delenv("MEMK_PROFILE", raising=False)
    monkeypatch.delenv("MEMK_EMBEDDER", raising=False)
    monkeypatch.setattr(embedder_mod, "_DEFAULT_EMBEDDER", None)
    monkeypatch.setattr(embedder_mod, "_DEFAULT_PIPELINE", None)

    profile = get_performance_profile()
    embedder = embedder_mod.get_default_embedder()

    assert profile.name == "lite"
    assert profile.default_embedder == "hashing"
    assert isinstance(embedder, embedder_mod.HashingEmbedder)


def test_fts_candidate_search_indexes_memory_and_facts(monkeypatch):
    monkeypatch.setenv("MEMK_PROFILE", "lite")
    db = MemoryDB(_db_path("fts"))
    db.init_db()

    memory_id = db.insert_memory("billing stripe token fix belongs in middleware")
    fact_id = db.insert_fact("billing", "uses", "stripe token")

    memory_hits = db.search_memory_fts("stripe token", limit=10)
    fact_hits = db.search_facts_fts("stripe token", limit=10)

    assert any(row["id"] == memory_id for row in memory_hits)
    assert any(row["id"] == fact_id for row in fact_hits)
    assert db.get_stats()["performance_profile"] == "lite"


def test_candidate_first_retriever_does_not_require_ram_index(monkeypatch):
    monkeypatch.setenv("MEMK_PROFILE", "lite")
    db = MemoryDB(_db_path("candidate"))
    db.init_db()
    db.insert_memory("auth token fix: session middleware accepts null token locally")

    retriever = CandidateFirstRetriever(db, candidate_limit=20, track_access=False)
    results = retriever.retrieve("auth token fix", limit=3)

    assert results
    assert results[0].item_type == "memory"
    assert "auth token fix" in results[0].content


def test_lite_runtime_skips_ram_index_graph_and_worker_threads(monkeypatch):
    import memk.core.embedder as embedder_mod
    from memk.core.runtime import RuntimeManager
    from memk.core.service import MemoryKernelService

    monkeypatch.setenv("MEMK_PROFILE", "lite")
    monkeypatch.setenv("MEMK_INDEX_MODE", "sqlite")
    monkeypatch.delenv("MEMK_GRAPH", raising=False)
    monkeypatch.delenv("MEMK_SPACY", raising=False)
    monkeypatch.setattr(embedder_mod, "_DEFAULT_EMBEDDER", None)
    monkeypatch.setattr(embedder_mod, "_DEFAULT_PIPELINE", None)
    RuntimeManager._instance = None

    service = MemoryKernelService(allow_direct_writes=True)
    runtime = service.global_runtime.get_workspace_runtime(
        "lite-test",
        db_path=_db_path("runtime"),
    )

    assert runtime.profile.name == "lite"
    assert runtime.index is None
    assert runtime.graph_index is None
    assert runtime.jobs._workers == []

    result = asyncio.run(service.add_memory(
        "auth token fix: session middleware accepts null token locally",
        importance=0.9,
        workspace_id="lite-test",
    ))
    search = asyncio.run(service.search(
        "session middleware null token",
        limit=3,
        workspace_id="lite-test",
    ))

    assert result["id"]
    assert search["results"]
    assert runtime.jobs._workers != []
    runtime.jobs.shutdown()
    RuntimeManager._instance = None
