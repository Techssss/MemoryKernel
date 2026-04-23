"""
memk.watcher.file_watcher
==========================
File watching infrastructure for real-time workspace change detection.

Features:
- Debounced event batching (2s window)
- Smart ignore rules (git, node_modules, binaries, etc.)
- Generation bumping on relevant changes
- Low CPU overhead with watchdog
- Workspace isolation

Phase 4B: MVP - metadata-only, no semantic parsing
"""

import logging
import time
import threading
from pathlib import Path
from typing import Set, Dict, Callable, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None
    FileSystemEvent = None

logger = logging.getLogger("memk.watcher")

# Configuration
DEBOUNCE_WINDOW_SEC = 2.0
IGNORE_PATTERNS = {
    ".git", ".memk", "node_modules", "__pycache__", ".pytest_cache",
    "venv", ".venv", "dist", "build", ".egg-info", ".tox",
    ".mypy_cache", ".ruff_cache", ".coverage", "htmlcov",
}

IGNORE_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".db", ".db-wal", ".db-shm", ".sqlite",
}

WATCH_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".sh", ".bash", ".zsh", ".fish",
}

@dataclass
class FileChangeEvent:
    """Normalized file change event."""
    path: str
    event_type: str
    timestamp: float = field(default_factory=time.time)
    
    def __hash__(self):
        return hash((self.path, self.event_type))

@dataclass
class WatcherStats:
    """Telemetry for watcher performance."""
    total_events: int = 0
    filtered_events: int = 0
    batched_events: int = 0
    generation_bumps: int = 0
    start_time: float = field(default_factory=time.time)
    
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
