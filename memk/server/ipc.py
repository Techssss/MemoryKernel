import json
import logging
import threading
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict
from memk.core.service import MemoryKernelService

logger = logging.getLogger("memk.transport")

class TransportServer(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

class IPCRequestProcessor:
    """Dispatches binary/json IPC requests to the core service."""
    def __init__(self, service: MemoryKernelService):
        self.service = service

    def process_raw(self, payload: str) -> str:
        try:
            req = json.loads(payload)
            action = req.get("action")
            args = req.get("args", {})
            
            result = None
            if action == "search":
                result = asyncio.run(self.service.search(**args))
            elif action == "add":
                result = asyncio.run(self.service.add_memory(**args))
            elif action == "context":
                result = asyncio.run(self.service.build_context(**args))
            elif action == "doctor":
                result = self.service.get_diagnostics()
            else:
                return json.dumps({"error": "unknown_action"})

            return json.dumps({"status": "ok", "data": result})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

class SimpleSocketServer(TransportServer):
    """
    Placeholder for a true Unix Domain Socket or Named Pipe server.
    Low-overhead IPC would use this instead of FastAPI/Uvicorn.
    """
    def __init__(self, service: MemoryKernelService, path: str = "/tmp/memk.sock"):
        self.service = service
        self.processor = IPCRequestProcessor(service)
        self.path = path
        self._running = False

    def start(self):
        logger.info(f"IPC Server (Socket) starting at {self.path}...")
        # In a real implementation, would use socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # and start a listening loop in a background thread.
        self._running = True

    def stop(self):
        self._running = False
        logger.info("IPC Server stopped.")
