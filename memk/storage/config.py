"""
memk.storage.config
===================
SQLite configuration and optimization for production.

Configures WAL mode, pragmas, and connection settings.
"""

import sqlite3
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("memk.storage.config")


@dataclass
class DatabaseConfig:
    """SQLite configuration settings."""
    
    # WAL mode for concurrent access
    journal_mode: str = "WAL"
    
    # Synchronous mode (FULL, NORMAL, OFF)
    # NORMAL is good balance of safety and speed
    synchronous: str = "NORMAL"
    
    # Cache size in KB (negative = KB, positive = pages)
    # -64000 = 64MB cache
    cache_size: int = -64000
    
    # Store temp tables in memory
    temp_store: str = "MEMORY"
    
    # Memory-mapped I/O size (256MB)
    mmap_size: int = 268435456
    
    # WAL autocheckpoint (pages)
    # Checkpoint when WAL reaches 1000 pages (~4MB)
    wal_autocheckpoint: int = 1000
    
    # Busy timeout (ms)
    # How long to wait if database is locked
    busy_timeout: int = 5000


def configure_connection(conn: sqlite3.Connection, config: Optional[DatabaseConfig] = None):
    """
    Configure a SQLite connection with production settings.
    
    Args:
        conn: SQLite connection to configure
        config: Configuration settings (uses defaults if None)
    """
    if config is None:
        from memk.core.profile import get_performance_profile

        profile = get_performance_profile()
        config = DatabaseConfig(
            cache_size=profile.sqlite_cache_size,
            mmap_size=profile.sqlite_mmap_size,
        )
    
    try:
        # Enable WAL mode
        result = conn.execute(f"PRAGMA journal_mode = {config.journal_mode}").fetchone()
        logger.info(f"Journal mode: {result[0]}")
        
        # Set synchronous mode
        conn.execute(f"PRAGMA synchronous = {config.synchronous}")
        
        # Set cache size
        conn.execute(f"PRAGMA cache_size = {config.cache_size}")
        
        # Set temp store
        conn.execute(f"PRAGMA temp_store = {config.temp_store}")
        
        # Set mmap size
        conn.execute(f"PRAGMA mmap_size = {config.mmap_size}")
        
        # Set WAL autocheckpoint
        conn.execute(f"PRAGMA wal_autocheckpoint = {config.wal_autocheckpoint}")
        
        # Set busy timeout (use PRAGMA, not attribute)
        conn.execute(f"PRAGMA busy_timeout = {config.busy_timeout}")
        
        logger.info("Database connection configured for production")
        
    except sqlite3.Error as e:
        logger.error(f"Failed to configure connection: {e}")
        raise


def get_database_info(conn: sqlite3.Connection) -> dict:
    """
    Get database configuration and status information.
    
    Returns:
        dict with configuration and status
    """
    info = {}
    
    try:
        # Get PRAGMA values
        pragmas = [
            "journal_mode",
            "synchronous",
            "cache_size",
            "temp_store",
            "mmap_size",
            "wal_autocheckpoint",
            "page_size",
            "page_count",
        ]
        
        for pragma in pragmas:
            result = conn.execute(f"PRAGMA {pragma}").fetchone()
            info[pragma] = result[0] if result else None
        
        # Calculate database size
        if info.get("page_size") and info.get("page_count"):
            info["size_bytes"] = info["page_size"] * info["page_count"]
            info["size_mb"] = round(info["size_bytes"] / (1024 * 1024), 2)
        
        # Get WAL info if in WAL mode
        if info.get("journal_mode") == "wal":
            wal_info = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            if wal_info:
                info["wal_busy"] = wal_info[0]
                info["wal_log_frames"] = wal_info[1]
                info["wal_checkpointed_frames"] = wal_info[2]
        
        return info
        
    except sqlite3.Error as e:
        logger.error(f"Failed to get database info: {e}")
        return {"error": str(e)}


def checkpoint_wal(conn: sqlite3.Connection, mode: str = "PASSIVE") -> dict:
    """
    Perform WAL checkpoint.
    
    Args:
        conn: SQLite connection
        mode: Checkpoint mode (PASSIVE, FULL, RESTART, TRUNCATE)
    
    Returns:
        dict with checkpoint results
    """
    try:
        result = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        
        return {
            "mode": mode,
            "busy": result[0],
            "log_frames": result[1],
            "checkpointed_frames": result[2],
            "success": result[0] == 0
        }
        
    except sqlite3.Error as e:
        logger.error(f"WAL checkpoint failed: {e}")
        return {"error": str(e), "success": False}


def optimize_database(conn: sqlite3.Connection) -> dict:
    """
    Run database optimization operations.
    
    Includes:
    - ANALYZE: Update query planner statistics
    - VACUUM: Reclaim unused space (if needed)
    
    Returns:
        dict with optimization results
    """
    results = {}
    
    try:
        # Run ANALYZE to update statistics
        logger.info("Running ANALYZE...")
        conn.execute("ANALYZE")
        results["analyze"] = "completed"
        
        # Check if VACUUM is needed
        info = get_database_info(conn)
        if info.get("page_count", 0) > 10000:  # Only for larger databases
            logger.info("Running VACUUM...")
            conn.execute("VACUUM")
            results["vacuum"] = "completed"
        else:
            results["vacuum"] = "skipped (database too small)"
        
        results["success"] = True
        return results
        
    except sqlite3.Error as e:
        logger.error(f"Database optimization failed: {e}")
        return {"error": str(e), "success": False}


def get_wal_status(db_path: str) -> dict:
    """
    Get WAL file status without opening connection.
    
    Returns:
        dict with WAL file information
    """
    import os
    
    wal_path = f"{db_path}-wal"
    shm_path = f"{db_path}-shm"
    
    status = {
        "db_exists": os.path.exists(db_path),
        "wal_exists": os.path.exists(wal_path),
        "shm_exists": os.path.exists(shm_path),
    }
    
    if status["db_exists"]:
        status["db_size_mb"] = round(os.path.getsize(db_path) / (1024 * 1024), 2)
    
    if status["wal_exists"]:
        status["wal_size_mb"] = round(os.path.getsize(wal_path) / (1024 * 1024), 2)
    
    if status["shm_exists"]:
        status["shm_size_kb"] = round(os.path.getsize(shm_path) / 1024, 2)
    
    return status

