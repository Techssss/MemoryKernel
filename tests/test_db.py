import os
import pytest
from memk.storage.db import MemoryDB, DatabaseError

@pytest.fixture
def test_db(tmp_path):
    # Use a temporary file-backed SQLite database for accurate connection handling
    db_file = tmp_path / "test_mem.db"
    db = MemoryDB(db_path=str(db_file))
    db.init_db()
    return db

def test_init_db_creates_table(test_db):
    """Test if init_db correctly creates the memories table."""
    conn = test_db._get_connection()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
    table = cursor.fetchone()
    assert table is not None
    assert table["name"] == "memories"

def test_insert_memory_success(test_db):
    """Test successful insertion of a memory."""
    mem_id = test_db.insert_memory("User prefers Python")
    assert mem_id is not None
    assert isinstance(mem_id, str)
    
    # Verify via direct query
    conn = test_db._get_connection()
    cursor = conn.execute("SELECT content FROM memories WHERE id = ?", (mem_id,))
    row = cursor.fetchone()
    assert row["content"] == "User prefers Python"

def test_insert_empty_memory_raises_error(test_db):
    """Test that inserting empty strings raises a ValueError."""
    with pytest.raises(ValueError):
        test_db.insert_memory("   ")

def test_search_memory_by_keyword(test_db):
    """Test keyword searching returns matching records."""
    test_db.insert_memory("User likes Python.")
    test_db.insert_memory("The backend is built with FastAPI.")
    test_db.insert_memory("Python is the default language for scripts.")
    
    # Search for "python" (case-insensitive in SQLite LIKE)
    results = test_db.search_memory("python")
    
    assert len(results) == 2
    assert any("likes Python" in r["content"] for r in results)
    assert any("default language for scripts" in r["content"] for r in results)

def test_search_memory_no_match(test_db):
    """Test searching with a non-existent keyword returns an empty list."""
    test_db.insert_memory("User likes Python.")
    results = test_db.search_memory("Java")
    assert len(results) == 0

def test_insert_and_search_facts(test_db):
    """Test inserting and successfully querying structured facts."""
    fact_id1 = test_db.insert_fact(subject="user", predicate="likes", obj="Python", confidence=0.9, importance=3)
    fact_id2 = test_db.insert_fact(subject="project", predicate="uses", obj="SQLite", confidence=1.0, importance=5)
    
    assert fact_id1 is not None
    assert fact_id2 is not None
    
    all_facts = test_db.search_facts()
    assert len(all_facts) == 2
    
    # Check ordering by created_at DESC (latest is first if inserted in very close succession, but standard auto-sorting tests might need precise mocks, so we just check content)
    subjects = [f["subject"] for f in all_facts]
    assert "user" in subjects
    assert "project" in subjects

def test_search_facts_by_subject(test_db):
    """Test filtering facts by specific subject."""
    test_db.insert_fact(subject="user", predicate="dislikes", obj="Java")
    test_db.insert_fact(subject="project", predicate="needs", obj="Docker")
    
    user_facts = test_db.search_facts(subject="user")
    assert len(user_facts) == 1
    assert user_facts[0]["predicate"] == "dislikes"
    assert user_facts[0]["object"] == "Java"
    
def test_insert_fact_validation_error(test_db):
    """Test that missing required fields raises ValueError."""
    with pytest.raises(ValueError):
        test_db.insert_fact(subject="", predicate="is", obj="Empty")
    
    with pytest.raises(ValueError):
        test_db.insert_fact(subject="user", predicate="", obj="Empty")

def test_fact_reconciliation(test_db):
    """Test that inserting a fact with identical subject/predicate marks the older fact as inactive."""
    test_db.insert_fact("user", "likes", "Java")
    
    # Should be 1 fact
    facts_before = test_db.search_facts(subject="user")
    assert len(facts_before) == 1
    assert facts_before[0]["object"] == "Java"
    
    # Send conflicting new fact
    test_db.insert_fact("user", "likes", "Kotlin")
    
    # Search should only return the new truth
    facts_after = test_db.search_facts(subject="user")
    assert len(facts_after) == 1
    assert facts_after[0]["object"] == "Kotlin"
    
    # Validate SQLite directly to ensure old fact still exists but is inactive
    with test_db._get_connection() as conn:
        cursor = conn.execute("SELECT object, is_active FROM facts WHERE subject='user' AND predicate='likes' ORDER BY created_at ASC")
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0]["object"] == "Java"
        assert rows[0]["is_active"] == 0
        assert rows[1]["object"] == "Kotlin"
        assert rows[1]["is_active"] == 1
