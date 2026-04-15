import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from .schema import WorkspaceManifest

MEMK_DIR = ".memk"
MANIFEST_FILE = "manifest.json"

class WorkspaceManager:
    def __init__(self, start_path: Optional[str] = None):
        self.start_path = Path(start_path or os.getcwd()).resolve()
        self.root = self.resolve_root(self.start_path)
        self.memk_path = self.root / MEMK_DIR
        self.manifest_path = self.memk_path / MANIFEST_FILE

    @staticmethod
    def resolve_root(path: Path) -> Path:
        """Find the git root or fallback to the current directory."""
        current = path
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
        # Fallback to the initial path if no git root found
        return path

    def is_initialized(self) -> bool:
        return self.manifest_path.exists()

    def init_workspace(self) -> WorkspaceManifest:
        """Create the .memk structure and manifest."""
        if not self.memk_path.exists():
            self.memk_path.mkdir(parents=True)
        
        # Create subdirectories for separation of concerns
        (self.memk_path / "state").mkdir(exist_ok=True)
        (self.memk_path / "index").mkdir(exist_ok=True)
        (self.memk_path / "cache").mkdir(exist_ok=True)
        (self.memk_path / "sync").mkdir(exist_ok=True)

        if self.manifest_path.exists():
            # Load existing
            return self.get_manifest()
        
        # New manifest
        manifest = WorkspaceManifest(workspace_root=str(self.root))
        self.save_manifest(manifest)
        
        # Create a default .gitignore inside .memk if it doesn't exist
        gitignore_path = self.memk_path / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, "w") as f:
                f.write("# Derived artifacts - do not commit\n")
                f.write("index/\n")
                f.write("cache/\n")
                f.write("sync/\n")
                f.write("state/*.db-wal\n")
                f.write("state/*.db-shm\n")

        return manifest

    def get_manifest(self) -> WorkspaceManifest:
        if not self.manifest_path.exists():
            raise RuntimeError("Workspace not initialized. Run 'memk init' first.")
        with open(self.manifest_path, "r") as f:
            data = json.load(f)
            return WorkspaceManifest(**data)

    def save_manifest(self, manifest: WorkspaceManifest):
        with open(self.manifest_path, "w") as f:
            json.dump(manifest.model_dump(), f, indent=4)

    def get_db_path(self) -> str:
        """Return the path to the SQLite database."""
        return str(self.memk_path / "state" / "state.db")

    def get_status_info(self) -> dict:
        initialized = self.is_initialized()
        info = {
            "initialized": initialized,
            "root": str(self.root),
            "memk_dir": str(self.memk_path),
        }
        if initialized:
            manifest = self.get_manifest()
            info.update(manifest.model_dump())
        return info

    def bump_generation(self) -> int:
        """
        Increment the generation counter atomically.
        Called whenever knowledge state changes (insert_memory, insert_fact).
        Returns the new generation number.
        """
        manifest = self.get_manifest()
        manifest.generation += 1
        self.save_manifest(manifest)
        return manifest.generation

    def get_generation(self) -> int:
        """Get current generation without bumping."""
        manifest = self.get_manifest()
        return manifest.generation
