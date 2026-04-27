"""
memk.sdk.client
===============
Python SDK client for MemoryKernel.

Simple, intuitive API for integrating memory into Python applications.
"""

import requests
import logging
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("memk.sdk")

DEFAULT_DAEMON_URL = os.getenv("MEMK_DAEMON_URL", "http://127.0.0.1:15301")


@dataclass
class MemoryItem:
    """A memory or fact retrieved from search."""
    item_type: str
    id: str
    content: str
    score: float
    importance: float
    confidence: float
    created_at: str
    access_count: int = 0
    decay_score: float = 1.0


@dataclass
class WorkspaceStatus:
    """Workspace status information."""
    workspace_id: str
    generation: int
    initialized: bool
    workspace_root: str
    total_memories: int
    total_facts: int
    watcher_running: bool = False


class MemoryKernel:
    """
    MemoryKernel SDK Client.
    
    Simple interface for adding and retrieving memories from a local workspace.
    
    Example:
        >>> mk = MemoryKernel()
        >>> mk.remember("User prefers TypeScript")
        >>> results = mk.search("What does user prefer?")
        >>> for r in results:
        ...     print(f"{r.score:.2f}: {r.content}")
    """
    
    def __init__(
        self,
        daemon_url: str = DEFAULT_DAEMON_URL,
        workspace_id: Optional[str] = None,
        auto_start_daemon: bool = False
    ):
        """
        Initialize MemoryKernel client.
        
        Args:
            daemon_url: URL of the memk daemon (default: http://127.0.0.1:15301)
            workspace_id: Workspace ID (auto-detected if not provided)
            auto_start_daemon: Attempt to start daemon if not running
        """
        self.daemon_url = daemon_url.rstrip("/")
        self.workspace_id = workspace_id
        self._generation: Optional[int] = None
        
        # Check daemon is running
        if not self._is_daemon_running():
            if auto_start_daemon:
                self._start_daemon()
            else:
                logger.warning(
                    f"Daemon not running at {daemon_url}. "
                    "Start with: memk serve"
                )
    
    def _is_daemon_running(self) -> bool:
        """Check if daemon is running."""
        try:
            resp = requests.get(f"{self.daemon_url}/v1/health", timeout=1)
            return resp.status_code == 200
        except:
            return False
    
    def _start_daemon(self):
        """Attempt to start daemon."""
        import subprocess
        try:
            subprocess.Popen(
                ["memk", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            import time
            time.sleep(2)  # Wait for startup
        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to daemon."""
        url = f"{self.daemon_url}{endpoint}"
        
        try:
            if method == "GET":
                resp = requests.get(url, **kwargs)
            elif method == "POST":
                resp = requests.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            resp.raise_for_status()
            return resp.json()
            
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to daemon at {self.daemon_url}. "
                "Start with: memk serve"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"API error: {e.response.text}")
    
    def remember(
        self,
        content: str,
        importance: float = 0.5,
        confidence: float = 1.0
    ) -> str:
        """
        Add a memory to the workspace.
        
        Args:
            content: Memory content to store
            importance: Priority/importance (0-1, default: 0.5)
            confidence: Confidence level (0-1, default: 1.0)
        
        Returns:
            Memory ID
        
        Example:
            >>> mk = MemoryKernel()
            >>> mem_id = mk.remember("User prefers dark mode", importance=0.8)
        """
        payload = {
            "content": content,
            "importance": importance,
            "confidence": confidence
        }
        
        if self.workspace_id:
            payload["workspace_id"] = self.workspace_id
        
        result = self._request("POST", "/v1/remember", json=payload)
        
        # Update generation
        self._generation = result["metadata"]["generation"]
        
        return result["data"]["id"]
    
    def search(
        self,
        query: str,
        limit: int = 10
    ) -> List[MemoryItem]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            limit: Maximum results to return (default: 10)
        
        Returns:
            List of MemoryItem objects ranked by relevance
        
        Example:
            >>> results = mk.search("What does user prefer?")
            >>> for r in results:
            ...     print(f"{r.score:.2f}: {r.content}")
        """
        payload = {
            "query": query,
            "limit": limit
        }
        
        if self.workspace_id:
            payload["workspace_id"] = self.workspace_id
        
        if self._generation is not None:
            payload["client_generation"] = self._generation
        
        result = self._request("POST", "/v1/search", json=payload)
        
        # Update generation
        self._generation = result["metadata"]["generation"]
        
        # Check for staleness
        if result["metadata"].get("stale_warning"):
            logger.warning(result["metadata"]["stale_warning"])
        
        # Convert to MemoryItem objects
        items = [MemoryItem(**item) for item in result["data"]["results"]]
        return items
    
    def context(
        self,
        query: str,
        max_chars: int = 500,
        threshold: float = 0.3
    ) -> str:
        """
        Build RAG context from relevant memories.
        
        Args:
            query: Context query
            max_chars: Maximum context length (default: 500)
            threshold: Relevance threshold (default: 0.3)
        
        Returns:
            Formatted context string
        
        Example:
            >>> context = mk.context("What should I know about the user?")
            >>> print(context)
        """
        payload = {
            "query": query,
            "max_chars": max_chars,
            "threshold": threshold
        }
        
        if self.workspace_id:
            payload["workspace_id"] = self.workspace_id
        
        if self._generation is not None:
            payload["client_generation"] = self._generation
        
        result = self._request("POST", "/v1/context", json=payload)
        
        # Update generation
        self._generation = result["metadata"]["generation"]
        
        # Check for staleness
        if result["metadata"].get("stale_warning"):
            logger.warning(result["metadata"]["stale_warning"])
        
        return result["data"]["context"]
    
    def status(self) -> WorkspaceStatus:
        """
        Get workspace status and statistics.
        
        Returns:
            WorkspaceStatus object
        
        Example:
            >>> status = mk.status()
            >>> print(f"Generation: {status.generation}")
            >>> print(f"Memories: {status.total_memories}")
        """
        params = {}
        if self.workspace_id:
            params["workspace_id"] = self.workspace_id
        
        result = self._request("GET", "/v1/status", params=params)
        
        data = result["data"]
        stats = data.get("stats", {})
        watcher = data.get("watcher", {})
        
        return WorkspaceStatus(
            workspace_id=data["workspace_id"],
            generation=data["generation"],
            initialized=data["initialized"],
            workspace_root=data["workspace_root"],
            total_memories=stats.get("total_memories", 0),
            total_facts=stats.get("total_active_facts", 0),
            watcher_running=watcher.get("running", False) if watcher else False
        )
    
    def ingest_git(
        self,
        limit: int = 50,
        since: Optional[str] = None,
        branch: str = "HEAD"
    ) -> Dict[str, Any]:
        """
        Ingest knowledge from Git commit history.
        
        Args:
            limit: Number of commits to ingest (default: 50)
            since: Only commits after this date (YYYY-MM-DD)
            branch: Git branch to ingest from (default: HEAD)
        
        Returns:
            Dict with ingested_count and categories
        
        Example:
            >>> result = mk.ingest_git(limit=100, since="2024-01-01")
            >>> print(f"Ingested {result['ingested_count']} memories")
        """
        payload = {
            "limit": limit,
            "branch": branch
        }
        
        if since:
            payload["since"] = since
        
        if self.workspace_id:
            payload["workspace_id"] = self.workspace_id
        
        result = self._request("POST", "/v1/ingest/git", json=payload)
        
        # Update generation
        self._generation = result["metadata"]["generation"]
        
        return result["data"]
    
    @property
    def generation(self) -> Optional[int]:
        """Get current generation number."""
        return self._generation

