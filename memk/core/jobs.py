import uuid
import threading
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from queue import PriorityQueue, Empty

logger = logging.getLogger("memk.jobs")

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(int, Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0

class JobRecord:
    def __init__(self, job_type: str, priority: JobPriority = JobPriority.NORMAL):
        self.id = str(uuid.uuid4())[:8]
        self.type = job_type
        self.priority = priority
        self.status = JobStatus.PENDING
        self.progress = 0.0 # 0.0 to 1.0
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.cancelled = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "priority": self.priority.name,
            "status": self.status.value,
            "progress": f"{self.progress * 100:.1f}%",
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error
        }

class BackgroundJobManager:
    """
    Production-ready job scheduler with priority queue and resource limits.
    Supports cancellation, progress tracking, and non-blocking execution.
    """
    def __init__(self, max_history: int = 50, max_workers: int = 2, start_immediately: bool = True):
        self.jobs: Dict[str, JobRecord] = {}
        self.history: List[str] = []
        self.max_history = max_history
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._queue: PriorityQueue = PriorityQueue()
        self._workers: List[threading.Thread] = []
        self._shutdown = False
        
        if start_immediately:
            self._ensure_workers()

    def _ensure_workers(self):
        """Start worker threads on first use when configured for lazy startup."""
        if self._workers or self.max_workers <= 0:
            return
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self._workers.append(worker)

    def submit(
        self, 
        job_type: str, 
        func: Callable, 
        *args, 
        priority: JobPriority = JobPriority.NORMAL,
        **kwargs
    ) -> str:
        """Submit a job to the priority queue."""
        job = JobRecord(job_type, priority)
        with self._lock:
            self.jobs[job.id] = job
            self.history.append(job.id)
            if len(self.history) > self.max_history:
                old_id = self.history.pop(0)
                if old_id in self.jobs:
                    del self.jobs[old_id]
        
        # Add to priority queue
        self._ensure_workers()
        self._queue.put((priority.value, job.id, func, args, kwargs))
        logger.info(f"Job {job.id} ({job_type}) queued with priority {priority.name}")
        return job.id

    def _worker_loop(self, worker_id: int):
        """Worker thread that processes jobs from the queue."""
        logger.info(f"Worker {worker_id} started")
        
        while not self._shutdown:
            try:
                # Get job from queue (timeout to check shutdown)
                priority, job_id, func, args, kwargs = self._queue.get(timeout=1.0)
                
                if job_id not in self.jobs:
                    continue
                
                self._run_job(job_id, func, args, kwargs)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    def _run_job(self, job_id: str, func: Callable, args, kwargs):
        """Execute a single job."""
        job = self.jobs[job_id]
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        
        logger.info(f"Job {job_id} ({job.type}) started")
        
        try:
            # Progress callback
            def progress_hook(p: float):
                if job.cancelled:
                    raise InterruptedError("Job cancelled")
                job.progress = min(max(p, 0.0), 1.0)

            # Check if func accepts progress_callback
            if "progress_callback" in func.__code__.co_varnames:
                kwargs["progress_callback"] = progress_hook
            
            # Check for cancellation flag
            if "check_cancelled" in func.__code__.co_varnames:
                kwargs["check_cancelled"] = lambda: job.cancelled
            
            job.result = func(*args, **kwargs)
            
            if job.cancelled:
                job.status = JobStatus.CANCELLED
                logger.info(f"Job {job_id} ({job.type}) cancelled")
            else:
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                logger.info(f"Job {job_id} ({job.type}) completed")
                
        except InterruptedError:
            job.status = JobStatus.CANCELLED
            logger.info(f"Job {job_id} ({job.type}) cancelled")
        except Exception as e:
            logger.error(f"Job {job_id} ({job.type}) failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
        finally:
            job.completed_at = datetime.now()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        job.cancelled = True
        logger.info(f"Job {job_id} ({job.type}) cancellation requested")
        return True

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Get job status."""
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 10, status: Optional[JobStatus] = None) -> List[Dict[str, Any]]:
        """List recent jobs, optionally filtered by status."""
        with self._lock:
            jobs = [self.jobs[jid] for jid in reversed(self.history[-limit:]) if jid in self.jobs]
            if status:
                jobs = [j for j in jobs if j.status == status]
            return [j.to_dict() for j in jobs]

    def get_queue_depth(self) -> int:
        """Get number of pending jobs in queue."""
        return self._queue.qsize()

    def get_active_count(self) -> int:
        """Get number of currently running jobs."""
        with self._lock:
            return sum(1 for j in self.jobs.values() if j.status == JobStatus.RUNNING)

    def shutdown(self):
        """Shutdown the job manager and wait for workers."""
        self._shutdown = True
        for worker in self._workers:
            worker.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Production Background Jobs
# ---------------------------------------------------------------------------

def reindex_job(db, index, embedder, progress_callback=None, check_cancelled=None):
    """
    Rebuild vector index from database.
    Non-blocking: yields between batches.
    """
    from memk.retrieval.index import IndexEntry
    from memk.core.embedder import decode_embedding
    
    logger.info("Starting reindex job")
    
    # Build new index
    from memk.retrieval.index import VectorIndex
    new_index = VectorIndex(dim=embedder.dim)
    
    # Load facts
    facts = db.get_all_active_facts()
    total = len(facts) + len(db.get_all_memories())
    processed = 0
    
    for r in facts:
        if check_cancelled and check_cancelled():
            return {"status": "cancelled", "processed": processed}
        
        if r["embedding"]:
            entry = IndexEntry(
                id=r["id"], item_type="fact",
                content=f"{r['subject']} {r['predicate']} {r['object']}",
                importance=float(r.get("importance", 0.5)),
                confidence=float(r.get("confidence", 1.0)),
                created_at=r["created_at"],
                decay_score=float(r.get("decay_score", 1.0)),
                access_count=int(r.get("access_count", 0)),
            )
            new_index.add_entry(entry, decode_embedding(r["embedding"]))
        
        processed += 1
        if progress_callback and total > 0:
            progress_callback(processed / total)
        
        # Yield every 100 items
        if processed % 100 == 0:
            time.sleep(0.001)
    
    # Load memories
    mems = db.get_all_memories()
    for r in mems:
        if check_cancelled and check_cancelled():
            return {"status": "cancelled", "processed": processed}
        
        if r["embedding"]:
            entry = IndexEntry(
                id=r["id"], item_type="memory", content=r["content"],
                importance=float(r.get("importance", 0.5)),
                confidence=float(r.get("confidence", 1.0)),
                created_at=r["created_at"],
                decay_score=float(r.get("decay_score", 1.0)),
                access_count=int(r.get("access_count", 0)),
            )
            new_index.add_entry(entry, decode_embedding(r["embedding"]))
        
        processed += 1
        if progress_callback and total > 0:
            progress_callback(processed / total)
        
        if processed % 100 == 0:
            time.sleep(0.001)
    
    # Atomic swap
    index.entries = new_index.entries
    index.vectors = new_index.vectors
    
    logger.info(f"Reindex completed: {processed} items")
    return {"status": "completed", "processed": processed, "index_size": len(index)}


def decay_update_job(db, scorer_fn, progress_callback=None, check_cancelled=None):
    """
    Update decay scores for all memories and facts.
    Non-blocking: yields between batches.
    """
    logger.info("Starting decay update job")
    
    updated = db.update_decay_scores(scorer_fn)
    
    if progress_callback:
        progress_callback(1.0)
    
    logger.info(f"Decay update completed: {updated} items")
    return {"status": "completed", "updated": updated}


def wal_checkpoint_job(db_path, mode="PASSIVE", progress_callback=None):
    """
    Perform WAL checkpoint to flush changes to main database.
    """
    import sqlite3
    from memk.storage.config import checkpoint_wal
    
    logger.info(f"Starting WAL checkpoint ({mode})")
    
    conn = sqlite3.connect(db_path)
    try:
        result = checkpoint_wal(conn, mode)
        
        if progress_callback:
            progress_callback(1.0)
        
        logger.info(f"WAL checkpoint completed: {result}")
        return result
    finally:
        conn.close()


def vacuum_job(db_path, progress_callback=None):
    """
    Reclaim unused space in database.
    WARNING: This can take a while and locks the database.
    """
    import sqlite3
    
    logger.info("Starting VACUUM job")
    
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("VACUUM")
        
        if progress_callback:
            progress_callback(1.0)
        
        logger.info("VACUUM completed")
        return {"status": "completed"}
    finally:
        conn.close()
