"""
Git Ingestion Demo
==================
Demonstrates Phase 4A Git history ingestion features.

This example shows:
1. How to ingest Git commit history
2. How rules match commits
3. How memories are generated
4. How to customize rules
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memk.ingestion.git_ingestor import (
    GitIngestor,
    IngestionRule,
    DEFAULT_RULES,
    ingest_git_history
)


def demo_basic_ingestion():
    """Demo 1: Basic Git ingestion."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Git Ingestion")
    print("="*60)
    
    # Get current repo path
    repo_path = Path(__file__).parent.parent
    
    print(f"\n→ Repository: {repo_path}")
    
    # Create ingestor
    ingestor = GitIngestor(repo_path=str(repo_path))
    
    # Check if Git repo
    if not ingestor.is_git_repo():
        print("❌ Not a Git repository!")
        return
    
    print("✓ Git repository detected")
    
    # Get recent commits
    print("\n→ Fetching last 10 commits...")
    hashes = ingestor.get_commit_hashes(limit=10)
    print(f"✓ Found {len(hashes)} commits")
    
    # Show first commit details
    if hashes:
        print("\n→ First commit details:")
        commit = ingestor.get_commit_metadata(hashes[0])
        print(f"  Hash: {commit.short_hash}")
        print(f"  Author: {commit.author}")
        print(f"  Date: {commit.timestamp}")
        print(f"  Message: {commit.summary}")
        print(f"  Files changed: {len(commit.files_changed)}")


def demo_rule_matching():
    """Demo 2: Rule matching."""
    print("\n" + "="*60)
    print("DEMO 2: Rule Matching")
    print("="*60)
    
    repo_path = Path(__file__).parent.parent
    ingestor = GitIngestor(repo_path=str(repo_path))
    
    if not ingestor.is_git_repo():
        print("❌ Not a Git repository!")
        return
    
    print(f"\n→ Default rules: {len(DEFAULT_RULES)}")
    
    # Show rules
    for rule in DEFAULT_RULES:
        print(f"\n  • {rule.name}")
        print(f"    Category: {rule.category}")
        print(f"    Keywords: {', '.join(rule.keywords[:3])}...")
        print(f"    Importance: {rule.importance}")
    
    # Get commits and match rules
    print("\n→ Matching rules against recent commits...")
    hashes = ingestor.get_commit_hashes(limit=20)
    
    matches = {}
    for hash_val in hashes:
        commit = ingestor.get_commit_metadata(hash_val)
        matched_rules = ingestor.match_rules(commit)
        
        if matched_rules:
            matches[commit.short_hash] = {
                "message": commit.summary,
                "rules": [r.name for r in matched_rules]
            }
    
    print(f"\n✓ Found {len(matches)} commits with rule matches:")
    for short_hash, data in list(matches.items())[:5]:
        print(f"\n  {short_hash}: {data['message'][:50]}...")
        print(f"    Rules: {', '.join(data['rules'])}")


def demo_memory_generation():
    """Demo 3: Memory generation."""
    print("\n" + "="*60)
    print("DEMO 3: Memory Generation")
    print("="*60)
    
    repo_path = Path(__file__).parent.parent
    
    print(f"\n→ Ingesting last 20 commits...")
    memories = ingest_git_history(
        repo_path=str(repo_path),
        limit=20
    )
    
    if not memories:
        print("❌ No memories generated (no commits matched rules)")
        return
    
    print(f"✓ Generated {len(memories)} memories")
    
    # Group by category
    by_category = {}
    for mem in memories:
        cat = mem["metadata"]["category"]
        by_category.setdefault(cat, []).append(mem)
    
    print("\n→ Memories by category:")
    for cat, mems in sorted(by_category.items()):
        print(f"\n  {cat.upper()} ({len(mems)} memories):")
        for mem in mems[:3]:  # Show first 3
            print(f"    • {mem['content'][:70]}...")
            print(f"      Importance: {mem['importance']}")


def demo_custom_rules():
    """Demo 4: Custom rules."""
    print("\n" + "="*60)
    print("DEMO 4: Custom Rules")
    print("="*60)
    
    # Define custom rules for this project
    custom_rules = [
        IngestionRule(
            name="phase_completion",
            category="decision",
            keywords=["phase", "complete", "implement"],
            importance=0.9
        ),
        IngestionRule(
            name="test_addition",
            category="convention",
            keywords=["test", "pytest"],
            file_patterns=["tests/"],
            importance=0.6
        ),
        IngestionRule(
            name="documentation_update",
            category="convention",
            keywords=["docs", "documentation", "readme"],
            file_patterns=[".md", "docs/"],
            importance=0.5
        ),
    ]
    
    print(f"\n→ Custom rules defined: {len(custom_rules)}")
    for rule in custom_rules:
        print(f"  • {rule.name} ({rule.category})")
    
    repo_path = Path(__file__).parent.parent
    ingestor = GitIngestor(repo_path=str(repo_path), rules=custom_rules)
    
    print("\n→ Ingesting with custom rules...")
    memories = ingestor.ingest_commits(limit=30)
    
    print(f"✓ Generated {len(memories)} memories with custom rules")
    
    # Show samples
    if memories:
        print("\n→ Sample memories:")
        for mem in memories[:5]:
            print(f"\n  [{mem['metadata']['category'].upper()}] {mem['content']}")
            print(f"    Rule: {mem['metadata']['rule']}")
            print(f"    Importance: {mem['importance']}")


def demo_duplicate_prevention():
    """Demo 5: Duplicate prevention."""
    print("\n" + "="*60)
    print("DEMO 5: Duplicate Prevention")
    print("="*60)
    
    repo_path = Path(__file__).parent.parent
    ingestor = GitIngestor(repo_path=str(repo_path))
    
    if not ingestor.is_git_repo():
        print("❌ Not a Git repository!")
        return
    
    # First ingestion
    print("\n→ First ingestion (limit=10)...")
    memories1 = ingestor.ingest_commits(limit=10)
    print(f"✓ Generated {len(memories1)} memories")
    print(f"  Processed commits: {len(ingestor.get_processed_commits())}")
    
    # Second ingestion (should skip all)
    print("\n→ Second ingestion (same commits)...")
    memories2 = ingestor.ingest_commits(limit=10)
    print(f"✓ Generated {len(memories2)} memories (should be 0)")
    print(f"  Processed commits: {len(ingestor.get_processed_commits())}")
    
    # Reset and try again
    print("\n→ Reset processed commits...")
    ingestor.reset_processed()
    print(f"  Processed commits: {len(ingestor.get_processed_commits())}")
    
    print("\n→ Third ingestion (after reset)...")
    memories3 = ingestor.ingest_commits(limit=10)
    print(f"✓ Generated {len(memories3)} memories (should match first run)")


def demo_filtering():
    """Demo 6: Commit filtering."""
    print("\n" + "="*60)
    print("DEMO 6: Commit Filtering")
    print("="*60)
    
    repo_path = Path(__file__).parent.parent
    ingestor = GitIngestor(repo_path=str(repo_path))
    
    if not ingestor.is_git_repo():
        print("❌ Not a Git repository!")
        return
    
    print("\n→ Analyzing commits...")
    hashes = ingestor.get_commit_hashes(limit=30)
    
    total = len(hashes)
    merge_count = 0
    no_match_count = 0
    ingested_count = 0
    
    for hash_val in hashes:
        commit = ingestor.get_commit_metadata(hash_val)
        
        if commit.message.lower().startswith("merge"):
            merge_count += 1
        elif not ingestor.match_rules(commit):
            no_match_count += 1
        else:
            ingested_count += 1
    
    print(f"\n✓ Analysis of {total} commits:")
    print(f"  • Merge commits (skipped): {merge_count}")
    print(f"  • No rule match (skipped): {no_match_count}")
    print(f"  • Would be ingested: {ingested_count}")
    print(f"  • Skip rate: {((merge_count + no_match_count) / total * 100):.1f}%")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Phase 4A: Git Ingestion Demo")
    print("="*60)
    
    try:
        demo_basic_ingestion()
        demo_rule_matching()
        demo_memory_generation()
        demo_custom_rules()
        demo_duplicate_prevention()
        demo_filtering()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
