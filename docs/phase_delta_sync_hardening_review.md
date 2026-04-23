# Phase: Delta Sync Hardening Review

## Overview
This document summarizes the audit and hardening of the Delta Synchronization lifecycle within MemoryKernel. The focus was on ensuring data integrity (LWW), operational observability, and robust metadata maintenance.

## 1. Stable Components (PASSED)
- **Merkle Consistency:** The `MerkleService` successfully rebuilds and refreshes bucket hashes. Deterministic bucket mapping ensures cross-device consistency.
- **Stress Recovery:** Stress tests prove that nodes eventually converge even after multiple rounds of updates and archival.
- **BLOB Integrity:** Embedding bytes are correctly serialized/deserialized during delta transport without corruption.
- **Archival Sync:** Soft-delete (archived) states are propagated correctly through the Delta protocol.

## 2. Hardened / Fixed (STABLE NOW)
- **LWW (Last Writer Wins) Implementation:**
    - **Issue:** Previously, incoming deltas used `INSERT OR REPLACE` blindly, which could overwrite newer local data with older remote data.
    - **Fix:** Implemented `ON CONFLICT(id) DO UPDATE ... WHERE excluded.version_hlc >= table.version_hlc`. This guarantees convergence towards the globally newest state.
- **Monotonic Checkpoints:**
    - **Issue:** Replica checkpoints could move backward if an older batch was applied late.
    - **Fix:** Added `WHERE excluded.last_applied_hlc >= last_applied_hlc` to `replica_checkpoint` upserts.
- **Orphan Metadata Cleanup:**
    - **Mechanism:** `cleanup_stale_row_hashes` now correctly identifies and deletes `row_hash` entries for physical rows that were deleted without going through the `oplog` interceptors.

## 3. Remaining Risks & Caveats
- **Offline Replica Oplog Growth:**
    - **Risk:** If a replica is offline indefinitely, the `oplog` will never prune beyond its last checkpoint.
    - **Recommendation:** Implement a "Maximum Retention TTL" (e.g., 30 days). Replicas lagging more than this should be forced to perform a full Merkle-based state sync instead of Oplog-based delta sync.
- **Merkle Bucket Collision Probability:**
    - **Risk:** 256 buckets is sufficient for thousands of items. For millions, the collision density in a single bucket might slow down sync.
    - **Fix:** Scale `num_buckets` dynamically or increase the default to 1024-4096 for production.
- **Race Condition in GC:**
    - **Risk:** Although SQLite handles transactions, a very long `REPLACE` batch during sync might conflict with `DELETE FROM oplog`.
    - **Mitigation:** Sync processes should use `IMMEDIATE` transactions.

## 4. Pre-Web-API Checklist
Before exposing this via a Web API (HTTP/gRPC):
1. [ ] **Authentication:** Verify `replica_id` ownership (prevent node impersonation).
2. [ ] **Compression:** Delta payloads can be large (embeddings); use Zstd/Gzip.
3. [ ] **Batch Size:** Limit delta fetch to ~500 items per request to avoid timeout.
4. [ ] **Rate Limiting:** Protect `fetch_delta_for_buckets` as it is computationally expensive (hash scanning).

## 5. Recommended Next Phase
**Phase: Hybrid Recovery & Conflict Resolution**
- **Merkle-Oplog Hybrid:** Use Oplog for fast sync (99% of cases) and Merkle only as a fallback/verification.
- **Interactive Conflict resolution:** Allow the user to "view" conflicts where LWW might not be enough (e.g. diverging decision notes).
- **Delta Compression:** Optimize storage of `oplog` by storing only diffs instead of full row replacements.
