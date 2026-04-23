# MemoryKernel — Architecture Notes: Hybrid Graph+Vector Integration

> Tài liệu kỹ thuật nội bộ. Mục đích: map codebase hiện tại, xác định
> extension points an toàn, và liệt kê checklist triển khai Phase 1.
>
> Không có code chức năng mới trong tài liệu này.

---

## 1. Bản đồ Module hiện tại

### 1.1 Tổng quan luồng dữ liệu (Current Flow)

```
USER INPUT (text)
  │
  ▼
MemoryKernelService.add_memory()          ← memk/core/service.py:64
  │
  ├─[1] embed(text)                       ← EmbeddingPipeline (core/embedder.py)
  │     (shared singleton, thread-pooled)
  │
  ├─[2] db.insert_memory()                ← MemoryDB (storage/db.py:148)
  │     (SQLite WAL, per-workspace)
  │
  ├─[3] index.add_entry()                 ← VectorIndex (retrieval/index.py:49)
  │     (NumPy flat, per-workspace RAM)
  │
  ├─[4] extractor.extract_facts(text)     ← RuleBasedExtractor (extraction/extractor.py:60)
  │     (1 regex pattern, sync on write path)
  │
  └─[5] for each fact:
        ├─ embed(fact_text)
        ├─ db.insert_fact()               ← MemoryDB (storage/db.py:230+)
        └─ index.add_entry(fact)
```

```
QUERY (text)
  │
  ▼
MemoryKernelService.search()              ← memk/core/service.py:157
  │
  ├─[1] cache check                       ← MemoryCacheManager (core/cache.py)
  │
  ├─[2] embed(query)                      ← EmbeddingPipeline
  │
  ├─[3] index.search(q_vec, top_k)        ← VectorIndex.search() — O(N) dot product
  │
  └─[4] retriever.rank_candidates()       ← ScoredRetriever (retrieval/retriever.py)
        │
        └─ MemoryScorer.score()           ← core/scorer.py:183
           5 chiều: vec + kw + imp + rec + conf
           fact_multiplier = 1.3x
```

### 1.2 Module Responsibility Map

| Module | File(s) | Trách nhiệm | Coupling |
|:---|:---|:---|:---|
| **Storage** | `storage/db.py` (641 dòng) | SQLite CRUD: `memories`, `facts`, `decisions`. WAL mode. | Chỉ nói SQL. Không biết về index/embedder. |
| **Storage Config** | `storage/config.py` | WAL pragmas, checkpoint, DB info. | Utility cho db.py. |
| **Migrations** | `storage/migrations.py` (319 dòng) | Forward-only schema versioning. V1→V4 hiện tại. | Chỉ phụ thuộc sqlite3. |
| **Extraction** | `extraction/extractor.py` (82 dòng) | `BaseExtractor` ABC + `RuleBasedExtractor`. 1 regex. | Protocol-based. Drop-in replaceable. |
| **Vector Index** | `retrieval/index.py` (147 dòng) | NumPy flat scan. `IndexEntry` dataclass. Per-workspace. | Không biết về DB. Pure RAM. |
| **Retriever** | `retrieval/retriever.py` (521 dòng) | 3 strategies: Keyword, Hybrid, **ScoredRetriever** (default). | Phụ thuộc DB + Index + Embedder + Scorer. |
| **Scorer** | `core/scorer.py` (308 dòng) | `ScoringWeights` (5D) + `MemoryScorer`. Stateless. | Zero dependency. Pure math. |
| **Service** | `core/service.py` (376 dòng) | Orchestrator: add_memory, search, build_context. | Phụ thuộc toàn bộ runtime. |
| **Runtime** | `core/runtime.py` (227 dòng) | `WorkspaceRuntime`: tạo DB+Index+Cache+Retriever+Extractor per workspace. | Central wiring point. |
| **DI Container** | `core/container.py` (331 dòng) | `DependencyContainer` + factory registration. | Alternative wiring (chưa dùng trong service.py). |
| **Protocols** | `core/protocols.py` (247 dòng) | 8 Protocol interfaces. | Contract layer. |
| **Background Jobs** | `core/jobs.py` (336 dòng) | `BackgroundJobManager` + PriorityQueue + worker threads. | Standalone. |
| **Forgetting** | `core/forgetting.py` (52 dòng) | Decay score calculation. Hot/warm/cold classification. | Standalone utility. |
| **Context Builder** | `context/builder.py` | Assemble ranked items → context string for LLM. | Reads RetrievedItem. |

### 1.3 Schema SQLite hiện tại (v4)

```sql
-- Core data
memories   (id, content, embedding, importance, confidence, access_count,
            last_accessed_at, decay_score, created_at)
facts      (id, subject, predicate, object, confidence, importance, embedding,
            access_count, last_accessed_at, decay_score, created_at, is_active)
decisions  (id, action, reason, used_fact_ids, created_at)

-- Infrastructure
schema_version    (version, applied_at, description)
background_jobs   (id, job_type, status, created_at, ...)
db_metadata       (key, value, updated_at)
```

### 1.4 Dependency Graph (hiện tại)

```
service.py ──→ runtime.py ──→ WorkspaceRuntime
                                 ├── MemoryDB        (storage/)
                                 ├── VectorIndex      (retrieval/index.py)
                                 ├── ScoredRetriever   (retrieval/retriever.py)
                                 │     └── MemoryScorer (core/scorer.py)
                                 ├── ContextBuilder    (context/builder.py)
                                 ├── RuleBasedExtractor (extraction/extractor.py)
                                 ├── MemoryCacheManager (core/cache.py)
                                 └── BackgroundJobManager (core/jobs.py)
```

---

## 2. Extension Points an toàn

### 2.1 Extractor — Drop-in replacement ✅ SAFE

**Current state:**
- `BaseExtractor` ABC đã tồn tại với `extract_facts(text) → List[StructuredFact]`.
- `RuleBasedExtractor` implement duy nhất, 1 pattern regex.
- `ExtractorProtocol` đã có trong `protocols.py`.
- Container factory `extractor_factory` trong `container.py:177`.

**Insertion point:**
- Tạo `SpaCyExtractor(BaseExtractor)` trong cùng file hoặc file mới.
- Swap factory trong `container.py:177` hoặc swap trực tiếp tại `runtime.py:60`.
- Không thay đổi public API (`extract_facts` signature giữ nguyên).

**Risk:** 🟢 ZERO — BaseExtractor là ABC, mọi consumer chỉ gọi `extract_facts()`.

### 2.2 Schema graph tables — Migration an toàn ✅ SAFE

**Current state:**
- Migration framework đã có, forward-only, version-tracked.
- Hiện ở V4. Thêm V5 là standard practice.

**Insertion point:**
- Tạo `migrate_v4_to_v5()` trong `migrations.py`.
- Thêm 4 bảng mới: `entity`, `mention`, `edge`, `consolidated_fact`.
- Cập nhật `CURRENT_SCHEMA_VERSION = 5`.
- Thêm CRUD methods vào `MemoryDB` class.

**Risk:** 🟢 LOW — Migration framework xử lý rollout. Bảng mới không ảnh hưởng bảng cũ.

**Schema đề xuất (Phase 1 tối thiểu):**
```sql
-- v5: Knowledge Graph sidecar
CREATE TABLE entity (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id    TEXT    NOT NULL,
    canonical_text  TEXT    NOT NULL,
    entity_type     TEXT,
    confidence      REAL    NOT NULL DEFAULT 0.5,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
CREATE UNIQUE INDEX idx_entity_canon ON entity(workspace_id, canonical_text, entity_type);

CREATE TABLE mention (
    memory_id       TEXT    NOT NULL,
    entity_id       INTEGER NOT NULL,
    role            TEXT,            -- 'subject' | 'object' | 'context'
    PRIMARY KEY (memory_id, entity_id, role)
) WITHOUT ROWID;

CREATE TABLE edge (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id    TEXT    NOT NULL,
    src_entity_id   INTEGER NOT NULL,
    rel_type        TEXT    NOT NULL,
    dst_entity_id   INTEGER NOT NULL,
    weight          REAL    NOT NULL DEFAULT 1.0,
    confidence      REAL    NOT NULL DEFAULT 0.5,
    source_memory_id TEXT   NOT NULL,
    created_at      TEXT    NOT NULL
);
CREATE INDEX idx_edge_src ON edge(workspace_id, src_entity_id);
CREATE INDEX idx_edge_dst ON edge(workspace_id, dst_entity_id);
```

### 2.3 GraphSidecar (RAM index) — Additive ✅ SAFE

**Current state:**
- `VectorIndex` trong `retrieval/index.py` chỉ giữ vectors + metadata.
- `WorkspaceRuntime` giữ instance `self.index`.
- Không có cấu trúc graph nào trong RAM.

**Insertion point:**
- Tạo file mới: `memk/core/graph_index.py`.
- Class `GraphSidecar` giữ: entity embeddings, CSR adjacency (edge_src, edge_dst, edge_w), entity→memory mapping.
- Nạp từ SQLite tables mới khi `WorkspaceRuntime.__init__()` gọi `_hydrate_index()`.
- Thêm `self.graph: Optional[GraphSidecar] = None` vào `WorkspaceRuntime`.

**Risk:** 🟢 LOW — Hoàn toàn additive. Graph là `Optional`, code cũ không bị ảnh hưởng nếu graph=None.

### 2.4 graph_score vào Scoring — Backward compatible ✅ SAFE

**Current state:**
- `ScoringWeights`: 5 trọng số `w1..w5` + `fact_multiplier`.
- `ScoreBreakdown`: 5 component scores.
- `MemoryScorer.score()`: tính `raw = w1*vec + w2*kw + w3*imp + w4*rec + w5*conf`.

**Insertion point — 2 lựa chọn:**

**Option A (khuyến nghị): Post-scoring mix trong retriever, KHÔNG sửa scorer.**
```python
# Trong ScoredRetriever hoặc GraphAwareRetriever mới:
final = (1 - graph_mix) * base_score + graph_mix * graph_bonus
```
- graph_mix = 0.0 → behavior cũ 100%.
- Không cần sửa ScoringWeights, ScoreBreakdown, MemoryScorer.

**Option B: Thêm w6 vào scorer.**
- Sửa `ScoringWeights` thêm `w6: float = 0.0`.
- Sửa `ScoreBreakdown` thêm `graph_score: float = 0.0`.
- Sửa `MemoryScorer.score()` thêm term.
- Backward compatible nếu `w6=0.0` default.
- Nhưng sửa nhiều file hơn và ảnh hưởng test suite.

**Risk:** Option A = 🟢 ZERO. Option B = 🟡 LOW (cần update tests).

### 2.5 Background extraction worker — Infrastructure đã có ✅ SAFE

**Current state:**
- `BackgroundJobManager` với PriorityQueue + daemon worker threads.
- `service.py:102-131`: extraction hiện tại chạy **sync trên write path**.
- Jobs framework đã hỗ trợ: `submit()`, progress tracking, cancellation.

**Insertion point:**
- Tạo job mới `graph_enrichment_job()` trong `core/jobs.py` hoặc file riêng.
- Submit via `runtime.jobs.submit("graph_enrich", ...)` sau khi write thành công.
- Worker đọc memories mới → chạy SpaCy → upsert entity/edge.

**Risk:** 🟢 LOW — Jobs framework đã production-ready. Chỉ thêm job type mới.

---

## 3. Phân tích rủi ro

### 3.1 Rủi ro KỸ THUẬT

| Rủi ro | Mức | Mitigation |
|:---|:---:|:---|
| spaCy model load chậm (~2-5s) | 🟡 | Lazy-load trong worker thread, không block startup. |
| Entity resolution sai (canonicalization) | 🟡 | Bắt đầu bằng lowercase+strip. Từ từ thêm fuzzy matching. |
| Graph propagation khuếch đại nhiễu | 🟡 | Hard prune `max_active=256`. Bắt đầu `graph_mix=0.15` thấp. |
| SQLite write contention thêm bảng | 🟢 | WAL mode handles concurrent reads. Graph writes là background. |
| RAM tăng do entity+edge arrays | 🟢 | Entity count << memory count. 10K entities ~40KB RAM. |

### 3.2 Rủi ro KIẾN TRÚC

| Rủi ro | Mức | Mitigation |
|:---|:---:|:---|
| Coupled graph vào retriever | 🟡 | Giữ GraphSidecar tách biệt. Retriever chỉ gọi `graph.get_bonus()`. |
| Breaking change nếu sửa scorer | 🟢 | Dùng Option A (post-scoring mix). Scorer không đổi. |
| Migration V5 trên DB production | 🟢 | Forward-only migration. Bảng mới, không alter bảng cũ. |
| spaCy là dependency nặng (~50MB) | 🟡 | Optional dependency. Import lazy. `pip install memk[graph]`. |

### 3.3 Những thứ KHÔNG nên làm ở Phase 1

- ❌ Không refactor `ScoringWeights` thêm w6 (dùng post-mix thay vì sửa scorer)
- ❌ Không thêm GLiNER (quá nặng, chỉ Phase 2 async path)
- ❌ Không sửa `IndexEntry` dataclass (thêm field sẽ break hydration)
- ❌ Không thay `VectorIndex.search()` signature
- ❌ Không thêm Merkle tree (Phase 3)
- ❌ Không thêm consolidation/archiving (Phase 2)

---

## 4. Luồng dữ liệu SAU Phase 1

```
USER INPUT (text)
  │
  ▼
service.add_memory()
  │
  ├─[1] embed(text) → vector
  ├─[2] db.insert_memory()        → memories table
  ├─[3] index.add_entry()         → RAM vector index
  ├─[4] extractor.extract_facts() → StructuredFact list    ← SpaCyExtractor (MỚI)
  │     (spaCy SVO + normalize)
  ├─[5] for each fact → db.insert_fact() + index
  └─[6] jobs.submit("graph_enrich", memory_id)             ← BACKGROUND (MỚI)
        │
        └─ Worker thread:
           ├─ Đọc memory text
           ├─ Chạy spaCy NER + SVO extraction
           ├─ Upsert entities → entity table (MỚI)
           ├─ Upsert mentions → mention table (MỚI)
           ├─ Upsert edges → edge table (MỚI)
           └─ Cập nhật GraphSidecar RAM arrays (MỚI)

QUERY (text)
  │
  ▼
service.search()
  │
  ├─[1] cache check
  ├─[2] embed(query)
  ├─[3] index.search(q_vec, top_k=64)   → base candidates
  ├─[4] retriever.rank_candidates()      → base_score (5D)
  ├─[5] graph.propagate(seed_entities)   → graph_bonus    ← MỚI
  └─[6] final = (1-α)*base + α*graph    → re-ranked      ← MỚI
```

---

## 5. Checklist triển khai Phase 1

> Thứ tự này đảm bảo mỗi bước test được độc lập trước khi đi tiếp.

### Phase 1A: Schema + Storage (không phụ thuộc spaCy)

- [ ] **M1.** Tạo `migrate_v4_to_v5()` trong `storage/migrations.py`
  - Tạo 3 bảng: `entity`, `mention`, `edge`
  - Cập nhật `CURRENT_SCHEMA_VERSION = 5`
  - Thêm vào `MIGRATIONS` list
- [ ] **M2.** Thêm CRUD methods vào `MemoryDB`:
  - `upsert_entity(workspace_id, canonical_text, entity_type) → entity_id`
  - `insert_mention(memory_id, entity_id, role)`
  - `insert_edge(workspace_id, src_id, rel_type, dst_id, weight, source_memory_id)`
  - `get_entities_for_memory(memory_id) → List[dict]`
  - `get_edges_for_entity(entity_id) → List[dict]`
  - `get_all_entities(workspace_id) → List[dict]`
  - `get_all_edges(workspace_id) → List[dict]`
- [ ] **M3.** Test migration: chạy `init_db()` trên DB V4 → verify V5.
- [ ] **M4.** Test CRUD: insert entity → insert edge → query back.

### Phase 1B: GraphSidecar (RAM index)

- [ ] **M5.** Tạo `memk/core/graph_index.py`:
  - `GraphSidecar` class
  - fields: `entity_ids`, `edge_src`, `edge_dst`, `edge_w`, `entity_to_mem_idx`
  - method: `hydrate_from_db(db, workspace_id)` — load từ SQLite
  - method: `add_entity(entity_id)`, `add_edge(src, dst, weight)`
  - method: `propagate(seed_entity_ids, steps=2, alpha=0.2, max_active=256) → dict[mem_idx, float]`
- [ ] **M6.** Test GraphSidecar: tạo 5 entities + 4 edges → propagate → verify 2-hop activation.
- [ ] **M7.** Tích hợp vào `WorkspaceRuntime`:
  - Thêm `self.graph: Optional[GraphSidecar] = None`
  - Trong `_hydrate_index()`: nếu entity table có data → `self.graph = GraphSidecar(); self.graph.hydrate_from_db()`

### Phase 1C: SpaCyExtractor

- [ ] **M8.** Tạo `SpaCyExtractor(BaseExtractor)` trong `extraction/extractor.py`:
  - Lazy-load spaCy model (`en_core_web_sm`)
  - Sử dụng dependency parse → SVO extraction
  - Normalize subject/object → canonical text
  - Trả về `List[StructuredFact]` (cùng contract)
- [ ] **M9.** Test: "Sarah works at Google" → `StructuredFact(sarah, works_at, google)`.
- [ ] **M10.** Swap extractor:
  - `runtime.py:60` → `self.extractor = SpaCyExtractor()` (hoặc config-driven)
  - Giữ `RuleBasedExtractor` làm fallback nếu spaCy không cài.

### Phase 1D: Background graph enrichment

- [ ] **M11.** Tạo `graph_enrich_job(db, graph, memory_id)` trong `core/jobs.py`:
  - Đọc memory text
  - Chạy SpaCyExtractor → facts
  - Với mỗi fact: upsert entity(subject), upsert entity(object), insert edge, insert mention
  - Cập nhật GraphSidecar.add_entity/add_edge in-memory
- [ ] **M12.** Hook vào `service.py:add_memory()`:
  - Sau step [5] (persist facts), submit `graph_enrich` job.
  - Job chạy async, không block write response.

### Phase 1E: Graph-aware retrieval

- [ ] **M13.** Tạo retrieval helper trong retriever hoặc service:
  - Sau `rank_candidates()` trả base results.
  - Lấy entity_ids từ top-K memories (via mention table hoặc RAM).
  - Gọi `graph.propagate(seed_entities)` → `graph_bonus: dict[mem_idx, float]`.
  - Mix: `final = (1 - graph_mix) * base + graph_mix * graph_bonus`.
  - Re-sort.
- [ ] **M14.** Test end-to-end:
  - Insert "Team A manages Project X"
  - Insert "Team A is located in Building C"
  - Query "Where is Project X?"
  - Verify Building C appears in results (via 2-hop: Project X → Team A → Building C).

### Phase 1F: Safety + backward compatibility

- [ ] **M15.** Feature flag: `MEMK_GRAPH_ENABLED=1` env var. Tắt = behavior cũ 100%.
- [ ] **M16.** Chạy test suite hiện tại → verify 0 regressions.
- [ ] **M17.** Benchmark: 100K records, query latency < 20ms với graph enabled.

---

## 6. Dependencies Phase 1

```
# Bắt buộc
spacy>=3.7     # ~12MB cho en_core_web_sm pipeline
textacy>=0.13  # SVO triple extraction từ spaCy doc

# Model download (one-time)
python -m spacy download en_core_web_sm

# Optional (extras)
# pip install memk[graph]
```

**Không cần ở Phase 1:** GLiNER, ONNX Runtime, Qwen, SciPy sparse, FAISS.

---

## 7. File map sau Phase 1

```
memk/
├── core/
│   ├── graph_index.py     ← MỚI: GraphSidecar (RAM adjacency + propagation)
│   ├── jobs.py            ← SỬA: thêm graph_enrich_job()
│   ├── runtime.py         ← SỬA: thêm self.graph, hydrate graph
│   ├── scorer.py          ← KHÔNG ĐỔI
│   ├── service.py         ← SỬA: submit graph_enrich job, graph rerank
│   └── ...
├── extraction/
│   └── extractor.py       ← SỬA: thêm SpaCyExtractor class
├── retrieval/
│   ├── index.py           ← KHÔNG ĐỔI
│   └── retriever.py       ← SỬA NHẸ: graph_bonus mix (hoặc trong service.py)
├── storage/
│   ├── db.py              ← SỬA: thêm entity/edge CRUD methods
│   └── migrations.py      ← SỬA: thêm V5 migration
└── ...
```

**Tổng cộng file sửa: 5.** File mới: 1. Không xóa file nào.
