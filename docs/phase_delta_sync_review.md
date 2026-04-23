# Delta Sync Phase - Code Review & Stabilization Report

## 1. What Was Completed & Stabilized
- **Delta Sync Schema (V9)**: Integrated `oplog`, `row_hash`, and `merkle_bucket` tables, fully decoupling the synchronization topology from normal flat-file storage schemas.
- **HLC Vector Versioning**: Bound Global Hybrid Logical Clock (`HLC`) tracking onto internal CRUD components successfully.
- **Local Protocol Algorithms**: Delivered pure-logic `diff_buckets`, `fetch_delta_for_buckets`, and `apply_remote_delta` APIs within `memk/sync/protocol.py`. Tested perfectly across two isolated runtime boundaries.
- **Critical Fix (JSON Serialization of BLOBs)**: Caught an implicit crashing bug where `embedding` BLOB vectors retrieved from `sqlite3` return as pure `bytes`. The initial logic called `json.dumps()` resulting in a crash. We successfully implemented a `bytes -> hex()` pre-processor loop in `db.py`.
- **Critical Fix (Update Sync Tracking)**: Noticed that prior implementation of `archive_memory` wasn't generating sync logs. We attached HLC bump functionality and appended `_log_sync_operation` implicitly ensuring soft-deletion behaves identically across network replicas.

## 2. Identified Risks & Remaining Constraints
- **Race Condition in Merkle Rebuilds**: Merkle buckets currently reconstruct manually taking `.fetchall()` of the entire `row_hash` table. In an extremely tight, concurrent write loop, a `rebuild_buckets()` command invoked while another background thread is parsing `_log_sync_operation` might slice tree definitions in unpredictable fractions. Currently mitigated by SQLite internal Write Locks, but scaling out could limit the IO layer.
- **Hard Deletions (Oplog Gaps)**: Operation code path `DELETE` correctly disposes of `row_hash` vectors now. However, `MemoryKernel` conventionally relies completely on Soft-Deletes (`archived = 1`). A hard deletion across a network would propagate an empty payload if a device attempts an `apply_remote_delta` operation without a dedicated tombstone system. 
- **Storage Growth Rate**: The `oplog` tracks an indefinite history. A system pushing 50k rows per day into the workspace will infinitely inflate the `.db` size until periodic maintenance is configured. 

## 3. Actions for the Next Phase
1. **Garbage Collection (Oplog Pruning)**: Implement an idle timeout background daemon tasked with purging historic `oplog` entries whose `version_hlc` passes a global `minimum_replicated_hlc` threshold.
2. **True Network Topology**: Expose the primitives tested here over a lightweight local RPC (e.g., `FastAPI` endpoint `/sync/merkle` or TCP/WebSocket logic).
3. **Optimized Conflict Resolution Strategy**: Instead of indiscriminately forcing `INSERT OR REPLACE`, develop a true generic Delta-CRDT strategy ensuring concurrent writes on disjoint devices resolve identically without data truncation.
