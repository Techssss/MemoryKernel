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
    
    assert "User facts:" in ctx
    assert "  - user likes python" in ctx
    assert "Project facts:" in ctx
    assert "  - project uses sqlite" in ctx
    assert "Raw memories:" in ctx
    assert "  - I once tried to learn django" in ctx

def test_context_truncation_budget():
    builder = ContextBuilder(max_chars=85)
    items = [
        RetrievedItem(item_type="fact", id="1", content="user loves coding and python", created_at="2026", score=2.0),
        RetrievedItem(item_type="memory", id="2", content="this secondary extremely long memory should be entirely dropped", created_at="2026", score=1.0)
    ]
    
    ctx = builder.build_context(items)
    
    # User fact gets in, but raw memory is dropped completely because budget constraint hit
    assert "User facts:" in ctx
    assert "user loves coding and python" in ctx
    assert "this secondary extremely long memory" not in ctx

def test_context_prioritization():
    # Extremely tight budget (only 30 chars), just enough for 'User facts:' title and short fact
    builder = ContextBuilder(max_chars=40)
    items = [
        RetrievedItem(item_type="fact", id="1", content="user uses c", created_at="", score=2.0),
        RetrievedItem(item_type="memory", id="2", content="I think I might like java", created_at="", score=1.0)
    ]
    ctx = builder.build_context(items)
    
    assert "User facts:" in ctx
    assert "user uses c" in ctx
    assert "Raw memories" not in ctx  # The second section could not even open
