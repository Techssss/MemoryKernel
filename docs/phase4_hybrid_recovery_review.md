# Phase 4 — Hybrid Recovery Review

**Reviewer**: Antigravity  
**Date**: 2026-04-23  
**Scope**: replica stale detection, sync mode selector, oplog vs merkle hybrid recovery, conflict_record schema, conflict detection on apply, conflict resolution primitives

---

## 1. What Is Stable

### Replica Health & Mode Selection (`health.py`)
- `SyncState` enum is clean and exhaustive: FRESH / LAGGING / STALE / UNKNOWN.
- `get_replica_sync_state` correctly compares checkpoint HLC against oplog range boundaries.
- `choose_sync_mode` correctly maps each state to the appropriate strategy.
- Empty-oplog edge case handled (returns STALE, forces Merkle).

### Hybrid Sync Orchestrator (`hybrid.py`)
- `_decide_mode` uses the **source node's** oplog range (correct direction — "can the source serve me the logs I need?").
- Clean separation: decision → execution → result.
- Oplog path and Merkle path both update checkpoints correctly.

### Merkle Recovery (`recovery.py`)
- Root hash comparison as first gate avoids unnecessary work.
- Bucket diff → delta fetch → LWW apply pipeline is correct.
- Passes checkpoint through `remote_replica_id` so future syncs resume correctly.

### Conflict Record Schema (migration v11, v12)
- Schema is clean: `conflict_id`, `table_name`, `row_id`, versions, snapshots, status, resolution, timestamps.
- Two indexes cover the two primary access patterns (status filter, row lookup).
- `resolved_ts` added in separate migration (v12) — clean incremental approach.

### Conflict Detection (`ConflictDetector`)
- Pure-function rules with no side effects — easy to test and extend.
- Three-layer gating reduces false positives:
  - Rule 1 (concurrent divergent) is the prerequisite gate.
  - Rule 3 (text divergence) fires only inside concurrent gate.
  - Rule 2 (cross-state) fires only if text didn't already match.
- Same-HLC writes correctly excluded (not concurrent).

### Conflict Resolution (`resolver.py`)
- Correctly distinguishes "LWW already agreed with my choice" (no-op) vs "need to revert data".
- Double-resolve and nonexistent-ID both return `{ok: False}` gracefully.
- `resolved_ts` always populated on success.

### Test Coverage
| Module | Tests | Status |
|:---|---:|:---|
| `test_conflict_record.py` | 11 | PASS |
| `test_conflict_detection.py` | 13 | PASS |
| `test_conflict_resolver.py` | 8 | PASS |
| `test_hybrid_sync_lifecycle.py` | 1 | PASS |
| `test_merkle_recovery.py` | 2 | PASS |
| `test_replica_health.py` | 3 | PASS |
| **Total** | **38** | **All passing** |

---

## 2. Bugs Found and Fixed This Review

### BUG-1 — row_hash poisoned after LWW rejection (CRITICAL — FIXED)

**File**: `protocol.py`, `apply_remote_delta`  
**Symptom**: When a remote delta had `version_hlc < local`, LWW correctly rejected the write. But the `row_hash` was unconditionally updated using the **remote payload** instead of the actual DB state.

**Impact**: Merkle tree would hash a row that doesn't exist locally. Bucket hashes diverge. Next Merkle recovery fetches data unnecessarily. Infinite recovery loop possible.

**Fix**: After upsert, re-`SELECT` the actual row from DB and hash that instead of the incoming payload.

### BUG-2 — resolver _force_write_snapshot skipped row_hash (MEDIUM — FIXED)

**File**: `resolver.py`, `_force_write_snapshot`  
**Symptom**: When a conflict resolution overrides LWW (e.g. `keep_local` when LWW chose remote), the semantic row was correctly reverted but `row_hash` was stale.

**Impact**: Next Merkle sync would see a hash mismatch for this row. Unnecessary re-fetch. Potential re-overwrite of the user's manual resolution.

**Fix**: Added `row_hash` update inside `_force_write_snapshot`, atomically in the same connection.

### BUG-3 — bare except in get_delta_since (LOW — FIXED)

**File**: `db.py`, `get_delta_since`  
**Symptom**: `except:` silently swallowed all errors including `KeyboardInterrupt` and schema errors.

**Impact**: If a table referenced in oplog was dropped or renamed, the sync would silently skip rows with zero diagnostics.

**Fix**: Changed to `except Exception as e:` with `logger.warning`.

---

## 3. What Is Still Risky

### RISK-1 — Conflict detection not wired into Merkle recovery path

`HybridSyncService.sync_from_source` calls `apply_remote_delta` **without** `detect_conflicts=True`. Merkle recovery also calls `apply_remote_delta` without it. This means conflicts during full-state recovery are invisible.

**Severity**: Low (Merkle recovery is state-based, so "conflict" is less meaningful — the whole point is to converge). But for high-value user data, it would be worth recording.

**Recommendation**: Add `detect_conflicts` as a config option on `HybridSyncService` and pass it through.

### RISK-2 — Resolver snapshot may contain hex-encoded BLOBs

`_safe_json` hex-encodes `bytes` fields when creating the conflict record. When `_force_write_snapshot` later writes the snapshot back, it writes hex strings instead of raw bytes into BLOB columns (like `embedding`).

**Severity**: Medium — affects embedding vectors. The row would have a string "0a0b0c..." where it should have bytes.

**Recommendation**: Add a `_restore_blobs` helper that detects known BLOB columns and decodes hex strings back to bytes before writing. Not urgent until resolution is used in production.

### RISK-3 — No tombstone / soft-delete awareness

`get_delta_since` skips rows that no longer exist in the semantic table (the oplog says "this row changed" but `SELECT * FROM memories WHERE id = ?` returns nothing). This means **deletes are not propagated** through the oplog path.

**Severity**: Medium — a row deleted on node A will never be deleted on node B through oplog sync. Merkle recovery partially covers this (row won't be in remote's row_hash), but local-only rows survive.

**Recommendation**: Phase 5 should add soft-delete (`is_deleted` flag) or explicit DELETE entries in the delta format.

### RISK-4 — fetch_delta_for_buckets scans entire row_hash table

`protocol.py:63` does `SELECT table_name, row_id, hash_val FROM row_hash` (full table scan) then filters in Python. For large DBs (100K+ rows), this is O(N) per recovery.

**Severity**: Low — only triggers during Merkle recovery, which is already the "slow path".

**Recommendation**: Optimize with a `bucket_id` column on `row_hash` or push the modulo filter into SQL.

---

## 4. Backward Compatibility

| Component | Backward Compatible? | Notes |
|:---|:---|:---|
| Migration v11, v12 | YES | Additive only (new table + new column) |
| `apply_remote_delta` | YES | `detect_conflicts` defaults to `False` |
| `HybridSyncService` | YES | New code, no changes to existing callers |
| `ConflictResolver` | YES | New code, unused until explicitly called |
| `SyncProtocolNode` | YES | No signature changes to existing methods |

**No breaking changes detected.**

---

## 5. Before Exposing Web API

These items should be resolved before any sync endpoint is exposed over HTTP:

1. **Wire `detect_conflicts=True`** into the Hybrid orchestrator (at least as an opt-in flag).
2. **Fix BLOB restoration** in resolver `_force_write_snapshot` so embeddings survive round-trip through JSON snapshots.
3. **Add rate limiting / pagination** to `get_delta_since` — currently fetches all changes since a given HLC with no cap.
4. **Add soft-delete propagation** — without it, the API will never sync deletions correctly.
5. **Authentication / authorization** on any sync endpoint — sync payloads contain full row data.

---

## 6. Recommended Next Phase

### Phase 5 — Sync API Readiness

| Priority | Task | Reason |
|:---|:---|:---|
| P0 | Soft-delete flag + DELETE propagation | Without this, deletes don't sync |
| P0 | BLOB round-trip safety in resolver | Embedding corruption risk |
| P1 | `detect_conflicts` wired into hybrid flow | Observability gap |
| P1 | Pagination for `get_delta_since` | Unbounded response size |
| P2 | `bucket_id` column on `row_hash` | Performance at scale |
| P2 | Conflict auto-resolve policies | Reduce manual review burden |
| P3 | Sync HTTP endpoints | Actual remote sync capability |
| P3 | Conflict review CLI / UI | Let users see and resolve conflicts |
