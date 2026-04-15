"""
memk.watcher.file_watcher
==========================
File watching infrastructure for real-time workspace change detection.
"""

import logging
import time
import threading
from pathlib import Path
from typing import Dict, Callable, Optional, List
from dataclasses import dataclass, field

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

DEBOUNCE_WINDOW_SEC = 2.0
IGNORE_PATTERNS = {".git", ".memk", "node_modules", "__pycache__", ".pytest_cache", "venv", ".venv", "dist", "build"}
IGNORE_EXTENSIONS = {".pyc", ".pyo", ".so", ".dll", ".exe", ".jpg", ".png", ".mp4", ".zip", ".db"}
WATCH_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".md", ".txt", ".json", ".yaml", ".sh"}

@dataclass
class FileChangeEvent:
    path: str
    event_type: str
    timestamp: float = field(default_factory=time.time)
    def __hash__(self):
        return hash((self.path, self.event_type))

@dataclass
class WatcherStats:
    total_events: int = 0
    filtered_events: int = 0
    batched_events: int = 0
    generation_bumps: int = 0
    start_time: float = field(default_factory=time.time)
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

class MemkFileEventHandler(FileSystemEventHandler):
    def __init__(self, root_path: Path, on_batch_callback: Callable[[List[FileChangeEvent]], None]):
        super().__init__()
        self.root_path = root_path
        self.on_batch_callback = on_batch_callback
        self.stats = WatcherStats()
        self._pending_events: Dict[str, FileChangeEvent] = {}
        self._debounce_lock = threading.Lock()
        self._debounce_timer: Optional[threading.Timer] = None
    
    def _should_ignore(self, path: Path) -> bool:
        for part in path.parts:
            if part in IGNORE_PATTERNS:
                return True
        if path.suffix.lower() in IGNORE_EXTENSIONS:
            return True
        if path.suffix and path.suffix.lower() not in WATCH_EXTENSIONS:
            return True
        return False
    
    def _normalize_event(self, event: FileSystemEvent) -> Optional[FileChangeEvent]:
        try:
            path = Path(event.src_path)
            if event.is_directory or self._should_ignore(path):
                self.stats.filtered_events += 1
                return None
            try:
                rel_path = path.relative_to(self.root_path)
            except ValueError:
                rel_path = path
            return FileChangeEvent(path=str(rel_path), event_type=event.event_type)
        except Exception:
            return None
    
    def _schedule_batch_processing(self):
        with self._debounce_lock:
            if self._debounce_timer:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(DEBOUNCE_WINDOW_SEC, self._process_batch)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()
    
    def _process_batch(self):
        with self._debounce_lock:
            if not self._pending_events:
                return
            events = list(self._pending_events.values())
            self._pending_events.clear()
            self._debounce_timer = None
        self.stats.batched_events += len(events)
        try:
            self.on_batch_callback(events)
        except Exception as e:
            logger.error(f"Error in batch callback: {e}")
    
    def _add_to_batch(self, event: FileChangeEvent):
        with self._debounce_lock:
            self._pending_events[event.path] = event
            self._schedule_batch_processing()
    
    def on_any_event(self, event: FileSystemEvent):
        self.stats.total_events += 1
        normalized = self._normalize_event(event)
        if normalized:
            self._add_to_batch(normalized)

class FileWatcher:
    def __init__(self, workspace_root: str, on_change_callback: Callable[[List[FileChangeEvent]], None]):
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError("watchdog library not installed")
        self.workspace_root = Path(workspace_root).resolve()
        self.on_change_callback = on_change_callback
        self.event_handler = MemkFileEventHandler(self.workspace_root, self._handle_batch)
        self.observer = Observer()
        self._is_running = False
    
    def _handle_batch(self, events: List[FileChangeEvent]):
        if not events:
            return
        logger.info(f"Processing batch of {len(events)} file changes")
        try:
            self.on_change_callback(events)
            self.event_handler.stats.generation_bumps += 1
        except Exception as e:
            logger.error(f"Error in change callback: {e}")
    
    def start(self):
        if self._is_running:
            logger.warning("Watcher already running")
            return
        logger.info(f"Starting file watcher for: {self.workspace_root}")
        self.observer.schedule(self.event_handler, str(self.workspace_root), recursive=True)
        self.observer.start()
        self._is_running = True
    
    def stop(self):
        if not self._is_running:
            return
        logger.info("Stopping file watcher...")
        self.observer.stop()
        self.observer.join(timeout=5.0)
        if self.event_handler._debounce_timer:
            self.event_handler._debounce_timer.cancel()
            self.event_handler._process_batch()
        self._is_running = False
    
    def is_running(self) -> bool:
        return self._is_running
    
    def get_stats(self) -> Dict:
        stats = self.event_handler.stats
        return {
            "running": self._is_running,
            "uptime_seconds": stats.uptime_seconds() if self._is_running else 0,
            "total_events": stats.total_events,
            "filtered_events": stats.filtered_events,
            "batched_events": stats.batched_events,
            "generation_bumps": stats.generation_bumps,
            "pending_events": len(self.event_handler._pending_events),
        }

class WatcherService:
    def __init__(self, workspace_root: str, workspace_manager):
        self.workspace_root = workspace_root
        self.workspace_manager = workspace_manager
        self.watcher: Optional[FileWatcher] = None
        self._change_log: List[Dict] = []
        self._max_log_size = 100
    
    def _on_file_changes(self, events: List[FileChangeEvent]):
        if not events:
            return
        try:
            from datetime import datetime
            new_generation = self.workspace_manager.bump_generation()
            change_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "generation": new_generation,
                "event_count": len(events),
                "files": [{"path": e.path, "type": e.event_type} for e in events[:10]],
            }
            self._change_log.append(change_record)
            if len(self._change_log) > self._max_log_size:
                self._change_log = self._change_log[-self._max_log_size:]
            logger.info(f"File changes detected → Generation bumped to {new_generation} ({len(events)} files)")
        except Exception as e:
            logger.error(f"Failed to bump generation: {e}")
    
    def start(self):
        if self.watcher and self.watcher.is_running():
            raise RuntimeError("Watcher already running")
        self.watcher = FileWatcher(self.workspace_root, self._on_file_changes)
        self.watcher.start()
    
    def stop(self):
        if self.watcher:
            self.watcher.stop()
            self.watcher = None
    
    def is_running(self) -> bool:
        return self.watcher is not None and self.watcher.is_running()
    
    def get_status(self) -> Dict:
        if not self.watcher:
            return {"running": False, "workspace_root": str(self.workspace_root)}
        stats = self.watcher.get_stats()
        stats["workspace_root"] = str(self.workspace_root)
        stats["recent_changes"] = self._change_log[-10:]
        return stats
