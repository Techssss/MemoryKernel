import pytest
from memk.retrieval.retriever import RetrievedItem
from memk.context.builder import ContextBuilder

def test_context_builder_grouping():
    builder = ContextBuilder(max_chars=500)
    items = [
        RetrievedItem(item_type="fact", id="1", content="user likes python", created_at="2026", score=2.0),
        RetrievedItem(item_type="fact", id="2", content="project uses sqlite", created_at="2026", score=2.0),
        RetrievedItem(item_type="memory", id="3", content="I once tried to learn django", created_at="2026", score=1.0),
    ]
    
    ctx = builder.build_context(items)
    
    assert "[User Preferences]" in ctx
    assert "• user likes python" in ctx
    assert "[Stable Facts]" in ctx
    assert "• project uses sqlite" in ctx
    assert "[Recent Memories]" in ctx
    assert "→ I once tried to learn django" in ctx

def test_context_truncation_budget():
    builder = ContextBuilder(max_chars=120)  # Increased slightly to accommodate new titles
    items = [
        RetrievedItem(item_type="fact", id="1", content="user loves coding and python", created_at="2026", score=2.0),
        RetrievedItem(item_type="memory", id="2", content="this secondary extremely long memory should be entirely dropped", created_at="2026", score=1.0)
    ]
    
    ctx = builder.build_context(items)
    
    # User fact gets in (it is categorized as User Preference if "user" in subject)
    assert "[User Preferences]" in ctx
    assert "user loves coding and python" in ctx
    assert "this secondary extremely long memory" not in ctx

def test_context_prioritization():
    # Priority: User Preferences > Recent Memories > Stable Facts > Conflicts
    builder = ContextBuilder(max_chars=60)
    items = [
        RetrievedItem(item_type="fact", id="1", content="user uses c", created_at="2026", score=2.0),
        RetrievedItem(item_type="memory", id="2", content="Secondary thought", created_at="2026", score=1.0)
    ]
    ctx = builder.build_context(items)
    
    # User Preference has higher priority than Recent memory
    assert "[User Preferences]" in ctx
    assert "user uses c" in ctx
