import time
import threading
import uuid
import os

class HLClock:
    """
    Hybrid Logical Clock implementation for Multi-Device Sync.
    Generates monotonically increasing (hlc, node_id, seq) tuples.
    """
    def __init__(self, node_id: str = None):
        if node_id is None:
            # Fallback to a random host ID if not specified by workspace or global config
            self.node_id = os.environ.get("MEMK_NODE_ID", f"node_{uuid.uuid4().hex[:6]}")
        else:
            self.node_id = node_id
            
        self._lock = threading.Lock()
        self._last_hlc = 0
        self._seq = 0
        
    def next_version(self) -> tuple[int, str, int]:
        """Returns (version_hlc, version_node, version_seq)"""
        with self._lock:
            now = int(time.time() * 1000)
            if now > self._last_hlc:
                self._last_hlc = now
                self._seq = 0
            else:
                # Same millisecond or clock drifted backward -> increment sequence
                self._seq += 1
                
            return (self._last_hlc, self.node_id, self._seq)

# Global clock fallback
GLOBAL_HLC = HLClock()
