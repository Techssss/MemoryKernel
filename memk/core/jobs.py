import uuid
import threading
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

logger = logging.getLogger("memk.jobs")

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class JobRecord:
    def __init__(self, job_type: str):
        self.id = str(uuid.uuid4())[:8]
        self.type = job_type
        self.status = JobStatus.PENDING
        self.progress = 0.0 # 0.0 to 1.0
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value,
            "progress": f"{self.progress * 100:.1f}%",
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error
        }

class BackgroundJobManager:
    """Simple thread-based job runner for heavy MemoryKernel tasks."""
    def __init__(self, max_history: int = 50):
        self.jobs: Dict[str, JobRecord] = {}
        self.history: List[str] = []
        self.max_history = max_history
        self._lock = threading.Lock()

    def submit(self, job_type: str, func: Callable, *args, **kwargs) -> str:
        job = JobRecord(job_type)
        with self._lock:
            self.jobs[job.id] = job
            self.history.append(job.id)
            if len(self.history) > self.max_history:
                old_id = self.history.pop(0)
                del self.jobs[old_id]

        # Start thread
        thread = threading.Thread(target=self._run_job, args=(job.id, func, args, kwargs), daemon=True)
        thread.start()
        return job.id

    def _run_job(self, job_id: str, func: Callable, args, kwargs):
        job = self.jobs[job_id]
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        
        try:
            # We pass a progress_callback to the function if it supports it
            def progress_hook(p: float):
                job.progress = min(max(p, 0.0), 1.0)

            # Check if func accepts progress_callback
            if "progress_callback" in func.__code__.co_varnames:
                kwargs["progress_callback"] = progress_hook
            
            job.result = func(*args, **kwargs)
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
        except Exception as e:
            logger.error(f"Job {job_id} ({job.type}) failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
        finally:
            job.completed_at = datetime.now()

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return [self.jobs[jid].to_dict() for jid in reversed(self.history[-limit:])]
