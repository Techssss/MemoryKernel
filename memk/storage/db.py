import sqlite3
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass

class MemoryDB:
    def __init__(self, db_path: str = "mem.db"):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a configured SQLite connection."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database at {self.db_path}: {e}")
            raise DatabaseError(f"Connection failed: {e}") from e

    def init_db(self) -> None:
        """Initialize the database schema."""
        query_memories = '''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
        query_facts = '''
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                importance INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        '''
        query_decisions = '''
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                used_fact_ids TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
        try:
            with self._get_connection() as conn:
                conn.execute(query_memories)
                conn.execute(query_facts)
                conn.execute(query_decisions)
                # Migration safe upgrade
                try:
                    conn.execute("ALTER TABLE facts ADD COLUMN is_active INTEGER DEFAULT 1")
                except sqlite3.OperationalError:
                    pass
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise DatabaseError(f"Initialization failed: {e}") from e

    def insert_memory(self, content: str) -> str:
        """Insert a new memory record and return its UUID."""
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty.")
        
        mem_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        query = '''
            INSERT INTO memories (id, content, created_at)
            VALUES (?, ?, ?)
        '''
        try:
            with self._get_connection() as conn:
                conn.execute(query, (mem_id, content.strip(), created_at))
            return mem_id
        except sqlite3.Error as e:
            logger.error(f"Failed to insert memory: {e}")
            raise DatabaseError(f"Insertion failed: {e}") from e

    def search_memory(self, keyword: str) -> List[Dict[str, Any]]:
        """Search memories containing the keyword (case-insensitive)."""
        if not keyword:
            return []

        query = '''
            SELECT id, content, created_at
            FROM memories
            WHERE content LIKE ?
            ORDER BY created_at DESC
        '''
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(query, (f"%{keyword}%",))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to search memories: {e}")
            raise DatabaseError(f"Search failed: {e}") from e

    def insert_fact(self, subject: str, predicate: str, obj: str, confidence: float = 1.0, importance: int = 1) -> str:
        """Insert a new structured fact and return its UUID. Automatically reconciles by shadowing old facts with the same subject and predicate."""
        if not subject or not predicate or not obj:
            raise ValueError("Fact subject, predicate, and object cannot be empty.")
        
        fact_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        recon_query = '''
            UPDATE facts 
            SET is_active = 0 
            WHERE subject = ? AND predicate = ?
        '''
        
        insert_query = '''
            INSERT INTO facts (id, subject, predicate, object, confidence, importance, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        '''
        try:
            with self._get_connection() as conn:
                # 1. Archive older iterations of the same exact context node
                conn.execute(recon_query, (subject.strip(), predicate.strip()))
                # 2. Append new node as the active truth
                conn.execute(insert_query, (fact_id, subject.strip(), predicate.strip(), obj.strip(), confidence, importance, created_at))
            return fact_id
        except sqlite3.Error as e:
            logger.error(f"Failed to insert fact: {e}")
            raise DatabaseError(f"Fact insertion failed: {e}") from e

    def search_facts(self, subject: str = None, keyword: str = None) -> List[Dict[str, Any]]:
        """Search facts, optionally matching a specific subject or broader keyword. Excludes inactive reconciled facts."""
        query = 'SELECT * FROM facts WHERE is_active = 1'
        params = []
        
        if subject:
            query += ' AND subject = ?'
            params.append(subject)
            
        if keyword:
            query += ' AND (subject LIKE ? OR predicate LIKE ? OR object LIKE ?)'
            like_val = f"%{keyword}%"
            params.extend([like_val, like_val, like_val])
            
        query += ' ORDER BY created_at DESC'
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to search facts: {e}")
            raise DatabaseError(f"Fact search failed: {e}") from e

    def log_decision(self, action: str, reason: str, used_fact_ids: List[str] = None) -> str:
        """Log a system or agent decision for observability."""
        import json
        decision_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        ids_str = json.dumps(used_fact_ids) if used_fact_ids else "[]"
        
        query = '''
            INSERT INTO decisions (id, action, reason, used_fact_ids, created_at)
            VALUES (?, ?, ?, ?, ?)
        '''
        try:
            with self._get_connection() as conn:
                conn.execute(query, (decision_id, action, reason, ids_str, created_at))
            return decision_id
        except sqlite3.Error as e:
            logger.error(f"Failed to log decision: {e}")
            raise DatabaseError(f"Decision logging failed: {e}") from e

    def get_stats(self) -> Dict[str, Any]:
        """Fetch basic observability statistics."""
        try:
            with self._get_connection() as conn:
                cur = conn.execute("SELECT COUNT(*) FROM memories")
                total_memories = cur.fetchone()[0]
                
                cur = conn.execute("SELECT COUNT(*) FROM facts WHERE is_active = 1")
                total_active_facts = cur.fetchone()[0]
                
                return {
                    "total_memories": total_memories,
                    "total_active_facts": total_active_facts
                }
        except sqlite3.Error as e:
            raise DatabaseError(f"Stats lookup failed: {e}") from e
