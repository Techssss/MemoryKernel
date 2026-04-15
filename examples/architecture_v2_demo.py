"""
Architecture V2 Demo
====================
Demonstrates the new protocol-based architecture with dependency injection.

This example shows:
1. Using the DI container
2. Lazy loading components
3. Custom implementations
4. Testing with mocks
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional

# Import V2 architecture components
from memk.core.container import (
    DependencyContainer,
    ContainerConfig,
    get_container,
    reset_container,
)
from memk.core.runtime_v2 import (
    RuntimeManagerV2,
    WorkspaceRuntimeV2,
    get_runtime_v2,
)
from memk.core.protocols import (
    StorageProtocol,
    EmbedderProtocol,
)


# ---------------------------------------------------------------------------
# Example 1: Basic Usage with Default Configuration
# ---------------------------------------------------------------------------

def example_1_basic_usage():
    """Basic usage with default configuration."""
    print("\n" + "="*70)
    print("Example 1: Basic Usage with Default Configuration")
    print("="*70)
    
    # Get the global container (singleton)
    container = get_container()
    
    # Create runtime manager
    runtime_manager = RuntimeManagerV2(container)
    runtime_manager.initialize_global()
    
    # Get workspace runtime (lazy-loaded)
    workspace = runtime_manager.get_workspace_runtime("demo-project")
    
    print(f"✓ Workspace created: {workspace.workspace_id}")
    print(f"✓ Generation: {workspace.get_generation()}")
    
    # Components are lazy-loaded
    print(f"✓ DB loaded: {workspace._db is not None}")
    print(f"✓ Index loaded: {workspace._index is not None}")
    
    # Access triggers loading
    _ = workspace.db
    print(f"✓ DB loaded after access: {workspace._db is not None}")
    
    # Get diagnostics
    diag = workspace.get_diagnostics()
    print(f"\nDiagnostics:")
    print(f"  - Components loaded: {sum(diag['components_loaded'].values())}/7")
    print(f"  - Index entries: {diag['index_entries']}")


# ---------------------------------------------------------------------------
# Example 2: Custom Configuration
# ---------------------------------------------------------------------------

def example_2_custom_configuration():
    """Using custom configuration."""
    print("\n" + "="*70)
    print("Example 2: Custom Configuration")
    print("="*70)
    
    reset_container()
    
    # Create custom configuration
    config = ContainerConfig(
        embedder_model="all-MiniLM-L6-v2",
        embedder_dim=384,
        cache_maxsize=200,  # Larger cache
        cache_ttl_seconds=7200,  # 2 hours
        max_workers=4,  # More workers
        enable_pipeline=True,
    )
    
    print(f"✓ Custom config created:")
    print(f"  - Cache size: {config.cache_maxsize}")
    print(f"  - Cache TTL: {config.cache_ttl_seconds}s")
    print(f"  - Max workers: {config.max_workers}")
    
    # Create container with config
    container = DependencyContainer(config)
    
    # Get diagnostics
    diag = container.get_diagnostics()
    print(f"\nContainer diagnostics:")
    print(f"  - Embedder loaded: {diag['singletons']['embedder_loaded']}")
    print(f"  - Registered factories: {len(diag['factories']['registered'])}")


# ---------------------------------------------------------------------------
# Example 3: Custom Implementation (Mock Storage)
# ---------------------------------------------------------------------------

class CustomStorage:
    """Custom storage implementation for demo."""
    
    def __init__(self):
        self.data = {}
        print("    [CustomStorage] Initialized")
    
    def init_db(self) -> None:
        print("    [CustomStorage] Database initialized")
    
    def insert_memory(
        self,
        content: str,
        *,
        embedding: Optional[np.ndarray] = None,
        importance: float = 0.5,
        confidence: float = 1.0,
    ) -> str:
        mem_id = f"custom-{len(self.data)}"
        self.data[mem_id] = {
            "content": content,
            "importance": importance,
        }
        print(f"    [CustomStorage] Inserted: {mem_id}")
        return mem_id
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        return list(self.data.values())
    
    def get_all_active_facts(self) -> List[Dict[str, Any]]:
        return []
    
    def search_memory(self, keyword: str) -> List[Dict[str, Any]]:
        return [m for m in self.data.values() if keyword in m["content"]]
    
    def search_facts(self, subject=None, keyword=None) -> List[Dict[str, Any]]:
        return []
    
    def touch_memory(self, mem_id: str) -> None:
        pass
    
    def touch_fact(self, fact_id: str) -> None:
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        return {"total_memories": len(self.data)}
    
    def get_fact_conflicts(self, active_fact_ids: List[str]) -> List[Dict[str, Any]]:
        return []
    
    def get_state_counts(self, cold_th: float, warm_th: float) -> Dict[str, int]:
        return {"hot": 0, "warm": 0, "cold": 0}


def example_3_custom_implementation():
    """Using custom storage implementation."""
    print("\n" + "="*70)
    print("Example 3: Custom Implementation")
    print("="*70)
    
    reset_container()
    
    # Create container
    container = DependencyContainer()
    
    # Register custom storage factory
    def custom_storage_factory(workspace_id: str, **kwargs):
        print(f"  Creating custom storage for: {workspace_id}")
        return CustomStorage()
    
    container.register_factory("storage", custom_storage_factory)
    
    print("✓ Custom storage factory registered")
    
    # Create runtime with custom storage
    runtime_manager = RuntimeManagerV2(container)
    workspace = runtime_manager.get_workspace_runtime("custom-project")
    
    print(f"\n✓ Workspace created: {workspace.workspace_id}")
    
    # Use custom storage
    mem_id = workspace.db.insert_memory("Test with custom storage")
    print(f"\n✓ Memory inserted: {mem_id}")
    
    # Retrieve
    memories = workspace.db.get_all_memories()
    print(f"✓ Retrieved {len(memories)} memories")


# ---------------------------------------------------------------------------
# Example 4: Testing with Mocks
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Mock embedder for testing."""
    
    @property
    def dim(self) -> int:
        return 384
    
    def embed(self, text: str) -> np.ndarray:
        print(f"    [MockEmbedder] Embedding: {text[:30]}...")
        # Simple hash-based mock
        hash_val = hash(text) % 384
        vec = np.zeros(384, dtype=np.float32)
        vec[hash_val] = 1.0
        return vec
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed(t) for t in texts]


def example_4_testing_with_mocks():
    """Testing with mock implementations."""
    print("\n" + "="*70)
    print("Example 4: Testing with Mocks")
    print("="*70)
    
    reset_container()
    
    # Create container with mocks
    container = DependencyContainer()
    
    # Override embedder with mock
    mock_embedder = MockEmbedder()
    container.set_embedder(mock_embedder)
    
    print("✓ Mock embedder registered")
    
    # Register mock storage
    container.register_factory("storage", lambda wid, **kw: CustomStorage())
    
    print("✓ Mock storage registered")
    
    # Create runtime
    runtime_manager = RuntimeManagerV2(container)
    workspace = runtime_manager.get_workspace_runtime("test-project")
    
    print(f"\n✓ Test workspace created: {workspace.workspace_id}")
    
    # Test embedding
    embedder = container.get_embedder()
    vec = embedder.embed("Test embedding")
    
    print(f"\n✓ Embedding created: shape={vec.shape}, sum={vec.sum():.2f}")
    
    # Test storage
    mem_id = workspace.db.insert_memory("Test memory")
    print(f"✓ Memory inserted: {mem_id}")


# ---------------------------------------------------------------------------
# Example 5: Workspace Lifecycle Management
# ---------------------------------------------------------------------------

def example_5_workspace_lifecycle():
    """Workspace lifecycle and eviction."""
    print("\n" + "="*70)
    print("Example 5: Workspace Lifecycle Management")
    print("="*70)
    
    reset_container()
    
    container = get_container()
    runtime_manager = RuntimeManagerV2(container)
    
    # Create multiple workspaces
    ws1 = runtime_manager.get_workspace_runtime("project-1")
    ws2 = runtime_manager.get_workspace_runtime("project-2")
    ws3 = runtime_manager.get_workspace_runtime("project-3")
    
    print(f"✓ Created 3 workspaces")
    print(f"  - {ws1.workspace_id}")
    print(f"  - {ws2.workspace_id}")
    print(f"  - {ws3.workspace_id}")
    
    # Check active workspaces
    diag = runtime_manager.get_diagnostics()
    print(f"\n✓ Active workspaces: {diag['global']['total_workspaces_active']}")
    
    # Simulate idle workspace
    ws1.last_active = 0  # Set to old time
    
    print(f"\n✓ Simulating idle workspace: {ws1.workspace_id}")
    
    # Evict idle workspaces
    runtime_manager.evict_idle_workspaces(idle_seconds=1)
    
    # Check again
    diag = runtime_manager.get_diagnostics()
    print(f"✓ Active workspaces after eviction: {diag['global']['total_workspaces_active']}")
    print(f"  - Remaining: {list(diag['active_workspaces'].keys())}")


# ---------------------------------------------------------------------------
# Example 6: Container Diagnostics
# ---------------------------------------------------------------------------

def example_6_diagnostics():
    """Comprehensive diagnostics."""
    print("\n" + "="*70)
    print("Example 6: Container Diagnostics")
    print("="*70)
    
    reset_container()
    
    # Create configured container
    config = ContainerConfig(
        cache_maxsize=150,
        max_workers=3,
    )
    container = DependencyContainer(config)
    
    # Create some workspaces
    runtime_manager = RuntimeManagerV2(container)
    runtime_manager.initialize_global()
    
    ws1 = runtime_manager.get_workspace_runtime("project-1")
    ws2 = runtime_manager.get_workspace_runtime("project-2")
    
    # Trigger some loading
    _ = ws1.db
    _ = ws1.index
    _ = ws2.db
    
    # Get comprehensive diagnostics
    diag = runtime_manager.get_diagnostics()
    
    print("\n📊 Global Diagnostics:")
    print(f"  - Model load time: {diag['global']['model_load_time_ms']:.0f}ms")
    print(f"  - Active workspaces: {diag['global']['total_workspaces_active']}")
    
    print("\n📦 Container:")
    print(f"  - Embedder loaded: {diag['container']['singletons']['embedder_loaded']}")
    print(f"  - Pipeline loaded: {diag['container']['singletons']['pipeline_loaded']}")
    print(f"  - Registered factories: {len(diag['container']['factories']['registered'])}")
    
    print("\n🏢 Workspaces:")
    for wid, ws_diag in diag['active_workspaces'].items():
        print(f"  {wid}:")
        loaded = sum(ws_diag['components_loaded'].values())
        print(f"    - Components loaded: {loaded}/7")
        print(f"    - Index entries: {ws_diag['index_entries']}")
        print(f"    - Generation: {ws_diag['generation']}")


# ---------------------------------------------------------------------------
# Main Demo
# ---------------------------------------------------------------------------

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MemoryKernel Architecture V2 Demo")
    print("="*70)
    print("\nDemonstrating protocol-based architecture with dependency injection")
    
    try:
        example_1_basic_usage()
        example_2_custom_configuration()
        example_3_custom_implementation()
        example_4_testing_with_mocks()
        example_5_workspace_lifecycle()
        example_6_diagnostics()
        
        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
