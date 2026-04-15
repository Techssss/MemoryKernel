"""
memk.ingestion.git_ingestor
============================
Metadata-first Git history ingestion for MemoryKernel.

Phase 4A: Simple, safe, low-CPU ingestion of commit metadata.
No LLM, no deep semantic parsing - just pattern matching.
"""

import subprocess
import re
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class CommitMetadata(BaseModel):
    """Lightweight commit metadata extracted from Git."""
    commit_hash: str
    author: str
    timestamp: str  # ISO format
    message: str
    files_changed: List[str] = Field(default_factory=list)
    insertions: int = 0
    deletions: int = 0
    
    @property
    def short_hash(self) -> str:
        return self.commit_hash[:8]
    
    @property
    def summary(self) -> str:
        """First line of commit message."""
        return self.message.split('\n')[0].strip()


class IngestionRule(BaseModel):
    """Rule for converting commit metadata to memory."""
    name: str
    category: str  # fix, decision, convention, warning
    keywords: List[str]
    file_patterns: List[str] = Field(default_factory=list)
    importance: float = 0.5
    
    def matches(self, commit: CommitMetadata) -> bool:
        """Check if this rule matches the commit."""
        message_lower = commit.message.lower()
        
        # Check keywords in message
        keyword_match = any(kw.lower() in message_lower for kw in self.keywords)
        
        # Check file patterns if specified
        if self.file_patterns:
            file_match = any(
                any(pattern in f for pattern in self.file_patterns)
                for f in commit.files_changed
            )
            return keyword_match and file_match
        
        return keyword_match
    
    def generate_memory(self, commit: CommitMetadata) -> str:
        """Generate memory text from commit."""
        return (
            f"[{self.category.upper()}] {commit.summary} "
            f"(by {commit.author}, {commit.short_hash})"
        )


# ---------------------------------------------------------------------------
# Default Rules
# ---------------------------------------------------------------------------

DEFAULT_RULES = [
    # Bug Fixes
    IngestionRule(
        name="bug_fix",
        category="fix",
        keywords=["fix", "bug", "issue", "resolve", "patch", "hotfix"],
        importance=0.6
    ),
    
    # Architecture Decisions
    IngestionRule(
        name="architecture_decision",
        category="decision",
        keywords=["refactor", "redesign", "architecture", "migrate", "upgrade"],
        importance=0.8
    ),
    
    # Code Conventions
    IngestionRule(
        name="convention",
        category="convention",
        keywords=["style", "format", "lint", "convention", "standard"],
        file_patterns=[".eslintrc", ".prettierrc", "pyproject.toml", ".editorconfig"],
        importance=0.5
    ),
    
    # Breaking Changes / Warnings
    IngestionRule(
        name="breaking_change",
        category="warning",
        keywords=["breaking", "deprecated", "remove", "drop support"],
        importance=0.9
    ),
    
    # Feature Additions
    IngestionRule(
        name="feature",
        category="decision",
        keywords=["add", "implement", "feature", "support", "enable"],
        importance=0.7
    ),
    
    # Documentation
    IngestionRule(
        name="documentation",
        category="convention",
        keywords=["docs", "documentation", "readme", "comment"],
        file_patterns=["README", ".md", "docs/"],
        importance=0.4
    ),
    
    # Performance
    IngestionRule(
        name="performance",
        category="decision",
        keywords=["performance", "optimize", "speed", "cache", "faster"],
        importance=0.7
    ),
    
    # Security
    IngestionRule(
        name="security",
        category="warning",
        keywords=["security", "vulnerability", "cve", "exploit", "unsafe"],
        importance=0.9
    ),
]


# ---------------------------------------------------------------------------
# Git Ingestor
# ---------------------------------------------------------------------------

class GitIngestor:
    """
    Ingest knowledge from Git commit history.
    
    Phase 4A: Metadata-first approach
    - Parse commit metadata only
    - Use simple keyword matching
    - No LLM, no deep semantic analysis
    - Track processed commits to avoid duplicates
    """
    
    def __init__(
        self,
        repo_path: Optional[str] = None,
        rules: Optional[List[IngestionRule]] = None
    ):
        self.repo_path = Path(repo_path or ".").resolve()
        self.rules = rules or DEFAULT_RULES
        self._processed_commits: Set[str] = set()
    
    # ------------------------------------------------------------------
    # Git Operations
    # ------------------------------------------------------------------
    
    def _run_git(self, *args: str) -> str:
        """Run git command and return output."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path)] + list(args),
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e.stderr}")
            raise RuntimeError(f"Git command failed: {e.stderr}")
    
    def is_git_repo(self) -> bool:
        """Check if the path is a Git repository."""
        try:
            self._run_git("rev-parse", "--git-dir")
            return True
        except:
            return False
    
    def get_commit_hashes(
        self,
        limit: Optional[int] = None,
        since: Optional[str] = None,
        branch: str = "HEAD"
    ) -> List[str]:
        """Get list of commit hashes."""
        args = ["log", branch, "--format=%H"]
        
        if limit:
            args.append(f"-n{limit}")
        
        if since:
            args.append(f"--since={since}")
        
        output = self._run_git(*args)
        return [line.strip() for line in output.split('\n') if line.strip()]
    
    def get_commit_metadata(self, commit_hash: str) -> CommitMetadata:
        """Extract metadata for a single commit."""
        # Get commit info
        format_str = "%H%n%an%n%aI%n%B"
        info = self._run_git("show", "-s", f"--format={format_str}", commit_hash)
        lines = info.split('\n')
        
        hash_val = lines[0]
        author = lines[1]
        timestamp = lines[2]
        message = '\n'.join(lines[3:]).strip()
        
        # Get file stats
        try:
            stats = self._run_git("show", "--stat", "--format=", commit_hash)
            files_changed = []
            insertions = 0
            deletions = 0
            
            for line in stats.split('\n'):
                if '|' in line:
                    # Parse file change line: "path/to/file.py | 10 +++++-----"
                    parts = line.split('|')
                    if len(parts) >= 2:
                        file_path = parts[0].strip()
                        files_changed.append(file_path)
                        
                        # Parse insertions/deletions
                        stat_part = parts[1].strip()
                        nums = re.findall(r'\d+', stat_part)
                        if nums:
                            changes = int(nums[0])
                            plus_count = stat_part.count('+')
                            minus_count = stat_part.count('-')
                            if plus_count > 0:
                                insertions += plus_count
                            if minus_count > 0:
                                deletions += minus_count
        except:
            files_changed = []
            insertions = 0
            deletions = 0
        
        return CommitMetadata(
            commit_hash=hash_val,
            author=author,
            timestamp=timestamp,
            message=message,
            files_changed=files_changed,
            insertions=insertions,
            deletions=deletions
        )
    
    # ------------------------------------------------------------------
    # Rule Matching
    # ------------------------------------------------------------------
    
    def match_rules(self, commit: CommitMetadata) -> List[IngestionRule]:
        """Find all rules that match this commit."""
        return [rule for rule in self.rules if rule.matches(commit)]
    
    def should_ingest(self, commit: CommitMetadata) -> bool:
        """Determine if commit should be ingested."""
        # Skip if already processed
        if commit.commit_hash in self._processed_commits:
            return False
        
        # Skip merge commits (usually noise)
        if commit.message.lower().startswith("merge"):
            return False
        
        # Must match at least one rule
        return len(self.match_rules(commit)) > 0
    
    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    
    def ingest_commits(
        self,
        limit: Optional[int] = None,
        since: Optional[str] = None,
        branch: str = "HEAD"
    ) -> List[Dict[str, Any]]:
        """
        Ingest commits and return memory candidates.
        
        Returns list of dicts with:
        - content: Memory text
        - importance: Importance score
        - metadata: Commit metadata
        """
        if not self.is_git_repo():
            raise RuntimeError(f"Not a Git repository: {self.repo_path}")
        
        logger.info(f"Ingesting commits from {self.repo_path}")
        
        # Get commit hashes
        hashes = self.get_commit_hashes(limit=limit, since=since, branch=branch)
        logger.info(f"Found {len(hashes)} commits to process")
        
        memories = []
        processed_count = 0
        skipped_count = 0
        
        for commit_hash in hashes:
            try:
                # Get metadata
                commit = self.get_commit_metadata(commit_hash)
                
                # Check if should ingest
                if not self.should_ingest(commit):
                    skipped_count += 1
                    continue
                
                # Match rules
                matched_rules = self.match_rules(commit)
                
                # Generate memories (one per matched rule)
                for rule in matched_rules:
                    memory_text = rule.generate_memory(commit)
                    
                    memories.append({
                        "content": memory_text,
                        "importance": rule.importance,
                        "metadata": {
                            "source": "git",
                            "commit_hash": commit.commit_hash,
                            "commit_short": commit.short_hash,
                            "author": commit.author,
                            "timestamp": commit.timestamp,
                            "category": rule.category,
                            "rule": rule.name,
                        }
                    })
                
                # Mark as processed
                self._processed_commits.add(commit.commit_hash)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process commit {commit_hash[:8]}: {e}")
                continue
        
        logger.info(
            f"Ingestion complete: {processed_count} commits processed, "
            f"{skipped_count} skipped, {len(memories)} memories generated"
        )
        
        return memories
    
    def get_processed_commits(self) -> Set[str]:
        """Get set of already processed commit hashes."""
        return self._processed_commits.copy()
    
    def mark_processed(self, commit_hash: str):
        """Mark a commit as processed."""
        self._processed_commits.add(commit_hash)
    
    def reset_processed(self):
        """Clear processed commits tracking."""
        self._processed_commits.clear()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def ingest_git_history(
    repo_path: Optional[str] = None,
    limit: int = 50,
    since: Optional[str] = None,
    rules: Optional[List[IngestionRule]] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to ingest Git history.
    
    Args:
        repo_path: Path to Git repository (default: current directory)
        limit: Maximum number of commits to process
        since: Only commits after this date (e.g., "2024-01-01")
        rules: Custom ingestion rules (default: DEFAULT_RULES)
    
    Returns:
        List of memory candidates
    """
    ingestor = GitIngestor(repo_path=repo_path, rules=rules)
    return ingestor.ingest_commits(limit=limit, since=since)
