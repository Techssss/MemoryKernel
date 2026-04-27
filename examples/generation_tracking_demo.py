"""
Generation Tracking Demo
========================
Demonstrates Phase 3 generation-based consistency features.

This example shows:
1. How generation increments on writes
2. How to detect stale context
3. How cache invalidation works
4. Multi-workspace isolation
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path

# Set daemon mode for testing
os.environ["MEMK_DAEMON_MODE"] = "1"

from memk.workspace.manager import WorkspaceManager
from memk.core.service import MemoryKernelService
from memk.core.runtime import RuntimeManager


async def demo_basic_generation_tracking():
    """Demo 1: Basic generation tracking."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Generation Tracking")
    print("="*60)
    
    # Setup temporary workspace
    tmpdir = tempfile.mkdtemp()
    git_dir = Path(tmpdir) / ".git"
    git_dir.mkdir()
    
    try:
        os.chdir(tmpdir)
        ws_mgr = WorkspaceManager(start_path=tmpdir)
        manifest = ws_mgr.init_workspace()
        ws_id = manifest.brain_id
        
        print(f"\n✓ Workspace created: {ws_id}")
        print(f"  Initial generation: {ws_mgr.get_generation()}")
        
        # Create service
        RuntimeManager._instance = None
        service = MemoryKernelService(allow_direct_writes=True)
        
        # Add first memory
        print("\n→ Adding memory: 'User prefers Python'")
        result1 = await service.add_memory(
            "User prefers Python",
            workspace_id=ws_id
        )
        print(f"  Generation after write: {result1['metadata']['generation']}")
        print(f"  Memory ID: {result1['id']}")
        
        # Add second memory
        print("\n→ Adding memory: 'System uses PostgreSQL'")
        result2 = await service.add_memory(
            "System uses PostgreSQL",
            workspace_id=ws_id
        )
        print(f"  Generation after write: {result2['metadata']['generation']}")
        print(f"  Extracted facts: {len(result2['extracted_facts'])}")
        
        # Verify generation persisted
        print(f"\n✓ Final generation in manifest: {ws_mgr.get_generation()}")
        
    finally:
        os.chdir("..")
        shutil.rmtree(tmpdir, ignore_errors=True)


async def demo_stale_context_detection():
    """Demo 2: Stale context detection."""
    print("\n" + "="*60)
    print("DEMO 2: Stale Context Detection")
    print("="*60)
    
    tmpdir = tempfile.mkdtemp()
    git_dir = Path(tmpdir) / ".git"
    git_dir.mkdir()
    
    try:
        os.chdir(tmpdir)
        ws_mgr = WorkspaceManager(start_path=tmpdir)
        manifest = ws_mgr.init_workspace()
        ws_id = manifest.brain_id
        
        RuntimeManager._instance = None
        service = MemoryKernelService(allow_direct_writes=True)
        
        # Add initial data
        print("\n→ Adding initial memory")
        await service.add_memory("Initial state", workspace_id=ws_id)
        
        # Client reads at generation 1
        print("\n→ Client searches at generation 1")
        result1 = await service.search(
            "state",
            workspace_id=ws_id,
            client_generation=1
        )
        print(f"  Current generation: {result1['metadata']['generation']}")
        print(f"  Stale warning: {result1['metadata']['stale_warning']}")
        print(f"  ✓ Context is FRESH")
        
        # State changes
        print("\n→ System state changes (new memory added)")
        await service.add_memory("Updated state", workspace_id=ws_id)
        print(f"  New generation: {ws_mgr.get_generation()}")
        
        # Client tries to use old context
        print("\n→ Client searches with OLD generation (1)")
        result2 = await service.search(
            "state",
            workspace_id=ws_id,
            client_generation=1  # Old generation
        )
        print(f"  Current generation: {result2['metadata']['generation']}")
        print(f"  Stale warning: {result2['metadata']['stale_warning']}")
        print(f"  ⚠ Context is STALE!")
        
        # Client refreshes
        print("\n→ Client refreshes with CURRENT generation (2)")
        result3 = await service.search(
            "state",
            workspace_id=ws_id,
            client_generation=2  # Current generation
        )
        print(f"  Current generation: {result3['metadata']['generation']}")
        print(f"  Stale warning: {result3['metadata']['stale_warning']}")
        print(f"  ✓ Context is FRESH again")
        
    finally:
        os.chdir("..")
        shutil.rmtree(tmpdir, ignore_errors=True)


async def demo_cache_invalidation():
    """Demo 3: Cache invalidation on generation change."""
    print("\n" + "="*60)
    print("DEMO 3: Cache Invalidation")
    print("="*60)
    
    tmpdir = tempfile.mkdtemp()
    git_dir = Path(tmpdir) / ".git"
    git_dir.mkdir()
    
    try:
        os.chdir(tmpdir)
        ws_mgr = WorkspaceManager(start_path=tmpdir)
        manifest = ws_mgr.init_workspace()
        ws_id = manifest.brain_id
        
        RuntimeManager._instance = None
        service = MemoryKernelService(allow_direct_writes=True)
        
        # Add initial data
        print("\n→ Adding memory: 'Python is great'")
        await service.add_memory("Python is great", workspace_id=ws_id)
        
        # First search (cache miss)
        print("\n→ First search for 'Python'")
        result1 = await service.search("Python", workspace_id=ws_id)
        print(f"  Cache hit: {result1['metadata']['cache_hit']}")
        print(f"  Results: {len(result1['results'])}")
        
        # Second search (cache hit)
        print("\n→ Second search for 'Python' (should hit cache)")
        result2 = await service.search("Python", workspace_id=ws_id)
        print(f"  Cache hit: {result2['metadata']['cache_hit']}")
        print(f"  ✓ Cache is working!")
        
        # Add new memory (bumps generation, invalidates cache)
        print("\n→ Adding new memory: 'JavaScript is also good'")
        await service.add_memory("JavaScript is also good", workspace_id=ws_id)
        print(f"  New generation: {ws_mgr.get_generation()}")
        
        # Third search (cache miss due to invalidation)
        print("\n→ Third search for 'Python' (cache should be invalidated)")
        result3 = await service.search("Python", workspace_id=ws_id)
        print(f"  Cache hit: {result3['metadata']['cache_hit']}")
        print(f"  ✓ Cache was properly invalidated!")
        
    finally:
        os.chdir("..")
        shutil.rmtree(tmpdir, ignore_errors=True)


async def demo_multi_workspace_isolation():
    """Demo 4: Multi-workspace generation isolation."""
    print("\n" + "="*60)
    print("DEMO 4: Multi-Workspace Isolation")
    print("="*60)
    
    tmpdir1 = tempfile.mkdtemp()
    tmpdir2 = tempfile.mkdtemp()
    
    try:
        # Setup workspace 1
        git1 = Path(tmpdir1) / ".git"
        git1.mkdir()
        os.chdir(tmpdir1)
        ws_mgr1 = WorkspaceManager(start_path=tmpdir1)
        manifest1 = ws_mgr1.init_workspace()
        ws_id1 = manifest1.brain_id
        
        # Setup workspace 2
        git2 = Path(tmpdir2) / ".git"
        git2.mkdir()
        os.chdir(tmpdir2)
        ws_mgr2 = WorkspaceManager(start_path=tmpdir2)
        manifest2 = ws_mgr2.init_workspace()
        ws_id2 = manifest2.brain_id
        
        RuntimeManager._instance = None
        service = MemoryKernelService(allow_direct_writes=True)
        
        print(f"\n✓ Workspace 1: {ws_id1[:8]}...")
        print(f"  Generation: {ws_mgr1.get_generation()}")
        print(f"\n✓ Workspace 2: {ws_id2[:8]}...")
        print(f"  Generation: {ws_mgr2.get_generation()}")
        
        # Write to workspace 1
        print("\n→ Writing to Workspace 1")
        os.chdir(tmpdir1)
        await service.add_memory("WS1 memory", workspace_id=ws_id1)
        print(f"  WS1 generation: {ws_mgr1.get_generation()}")
        print(f"  WS2 generation: {ws_mgr2.get_generation()} (unchanged)")
        
        # Write to workspace 2
        print("\n→ Writing to Workspace 2")
        os.chdir(tmpdir2)
        await service.add_memory("WS2 memory", workspace_id=ws_id2)
        print(f"  WS1 generation: {ws_mgr1.get_generation()} (unchanged)")
        print(f"  WS2 generation: {ws_mgr2.get_generation()}")
        
        # Write again to workspace 1
        print("\n→ Writing again to Workspace 1")
        os.chdir(tmpdir1)
        await service.add_memory("WS1 second memory", workspace_id=ws_id1)
        print(f"  WS1 generation: {ws_mgr1.get_generation()}")
        print(f"  WS2 generation: {ws_mgr2.get_generation()} (unchanged)")
        
        print("\n✓ Workspaces are properly isolated!")
        
    finally:
        os.chdir("..")
        shutil.rmtree(tmpdir1, ignore_errors=True)
        shutil.rmtree(tmpdir2, ignore_errors=True)


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Phase 3: Generation Tracking Demo")
    print("="*60)
    
    await demo_basic_generation_tracking()
    await demo_stale_context_detection()
    await demo_cache_invalidation()
    await demo_multi_workspace_isolation()
    
    print("\n" + "="*60)
    print("All demos completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
