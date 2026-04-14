import pytest
from memk.storage.db import MemoryDB
from memk.retrieval.retriever import KeywordRetriever

@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_retrieval.db"
    db = MemoryDB(db_path=str(db_file))
    db.init_db()
    return db

@pytest.fixture
def retriever(test_db):
    return KeywordRetriever(test_db)

def test_retrieval_returns_facts_before_memories(test_db, retriever):
    test_db.insert_memory("User wants to build something with Python.")
    test_db.insert_fact("user", "loves", "Python")
    
    results = retriever.retrieve("Python")
    
    assert len(results) == 2
    # First result should be the fact
    assert results[0].item_type == "fact"
    assert "user loves Python" in results[0].content
    
    # Second result should be the raw memory
    assert results[1].item_type == "memory"
    assert "User wants to build" in results[1].content

def test_retrieval_newer_entries_first(test_db, retriever):
    # Use DISTINCT predicates so both facts are independently active
    # (same subject+predicate would trigger reconciliation → only latest stays active)
    test_db.insert_fact("user", "tried", "Go")
    test_db.insert_fact("user", "adopted", "Rust")

    results = retriever.retrieve("user")
    assert len(results) == 2
    # The second one inserted (Rust / adopted) should be ranked higher due to date sorting
    assert "Rust" in results[0].content
    assert "Go" in results[1].content

def test_empty_query_returns_empty(retriever):
    assert retriever.retrieve("") == []

def test_retrieve_across_subject_and_object(test_db, retriever):
    test_db.insert_fact("project", "runs on", "Docker")
    results = retriever.retrieve("Docker")
    assert len(results) == 1
    assert "Docker" in results[0].content
