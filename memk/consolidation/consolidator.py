import numpy as np
import logging
from typing import List, Dict, Any
from collections import defaultdict
import uuid
import json

from memk.core.runtime import WorkspaceRuntime

logger = logging.getLogger("memk.consolidation")

class ConsolidatorService:
    """
    Background worker service that groups highly similar, redundant raw memories 
    into consolidated canonical facts, keeping the vector space clean and scalable.
    """
    def __init__(self, runtime: WorkspaceRuntime, cosine_threshold: float = 0.90):
        self.runtime = runtime
        self.db = runtime.db
        self.cosine_threshold = cosine_threshold

    def get_candidate_memories(self, centroid_id: str = None) -> List[Dict[str, Any]]:
        """Fetch unarchived memories, optionally filtered by spatial centroid to scale."""
        query = "SELECT * FROM memories WHERE archived = 0"
        params = []
        if centroid_id:
            query += " AND (centroid_id = ? OR centroid_id IS NULL)"
            params.append(centroid_id)
            
        with self.db._get_connection() as conn:
            return [dict(r) for r in conn.execute(query, tuple(params)).fetchall()]

    def build_similarity_graph(self, candidates: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """O(N^2) similarity graph building using fast NumPy vectorization."""
        if len(candidates) < 2:
            return {}
            
        from memk.storage.db import _decode_blob
        
        embs = []
        valid_cands = []
        
        for c in candidates:
            if c.get("embedding"):
                try:
                    vec = _decode_blob(c["embedding"])
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        embs.append(vec / norm)
                        valid_cands.append(c)
                except Exception:
                    pass
                    
        if len(valid_cands) < 2:
            return {}
            
        emb_matrix = np.array(embs) # (N, D)
        sim_matrix = np.dot(emb_matrix, emb_matrix.T) # (N, N)
        
        adj = defaultdict(list)
        for i in range(len(valid_cands)):
            for j in range(i + 1, len(valid_cands)):
                if sim_matrix[i, j] >= self.cosine_threshold:
                    id_i, id_j = valid_cands[i]["id"], valid_cands[j]["id"]
                    adj[id_i].append(id_j)
                    adj[id_j].append(id_i)
                    
        return adj

    def extract_clusters(self, adj: Dict[str, List[str]], all_ids: List[str]) -> List[List[str]]:
        """Union-Find / Connected Components extraction."""
        visited = set()
        clusters = []
        
        for node in all_ids:
            if node not in visited and node in adj:
                comp = []
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        comp.append(curr)
                        for neighbor in adj.get(curr, []):
                            if neighbor not in visited:
                                stack.append(neighbor)
                if len(comp) > 1:
                    clusters.append(comp)
        return clusters

    def run_consolidation_job(self, centroid_id: str = None) -> int:
        """
        Execute consolidation pass on candidates. 
        Returns the number of created clusters/facts.
        """
        cands = self.get_candidate_memories(centroid_id)
        if len(cands) < 2:
            return 0
            
        cands_map = {c["id"]: c for c in cands}
        adj = self.build_similarity_graph(cands)
        clusters = self.extract_clusters(adj, list(cands_map.keys()))
        
        consolidated_count = 0
        
        from memk.storage.db import _utcnow
        
        for cluster_ids in clusters:
            if self._process_cluster(cluster_ids, cands_map, _utcnow()):
                consolidated_count += 1
                
        return consolidated_count

    def _process_cluster(self, cluster_ids: List[str], cands_map: Dict[str, Dict[str, Any]], ts: str) -> bool:
        """Merge memory cluster into a single kg_fact and archive originals."""
        items = [cands_map[cid] for cid in cluster_ids]
        
        # 1. Safety heuristics
        workspace_id = self.runtime.workspace_id
        
        # Anti-merge rule: Different distinct short topics might collide if vector aligns poorly
        # Usually handled by threshold > 0.90. We can add polarity checks here later.
        
        # 2. Extract Canonical Fact
        # FUTURE HOOK: Call Local LLM Summarizer (e.g. Qwen / Phi3)
        canonical_text = self._rule_based_summarize(items)
        
        avg_conf = sum(float(i.get("confidence", 1.0)) for i in items) / len(items)
        
        try:
            f_id = f"fact_{uuid.uuid4().hex[:8]}"
            summary_json = json.dumps({"source_memories": cluster_ids, "method": "rule_based"})
            
            with self.db._get_connection() as conn:
                conn.execute("""
                    INSERT INTO kg_fact (id, workspace_id, canonical_text, summary_json, confidence, created_ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (f_id, workspace_id, canonical_text, summary_json, avg_conf, ts))
                
            # Archive raw memories to decouple from hot vector search
            for cid in cluster_ids:
                self.db.archive_memory(cid)
                
            logger.info(f"[{workspace_id}] Consolidated {len(cluster_ids)} memories -> Fact {f_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to consolidate cluster: {e}")
            return False
            
    def _rule_based_summarize(self, items: List[Dict[str, Any]]) -> str:
        """
        Placeholder rule-based canonical text generator.
        Limitation: Will just pick the longest/most descriptive string until LLM pipeline is ready.
        """
        longest = max((i["content"] for i in items), key=len)
        return longest
