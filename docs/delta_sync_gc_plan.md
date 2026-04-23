# Bản Audit và Kế Hoạch Hardening Delta Sync (GC & Checkpointing)

## 1. Trạng Thái Hiện Tại (Current State)
Dựa vào quá trình audit mã nguồn, hệ thống Delta Sync hiện đang ở mức prototype cục bộ. Logic xoay quanh 3 bảng chính `oplog`, `row_hash`, và `merkle_bucket` với các đặc điểm:

- **Oplog Growth**: Bảng `oplog` ghi nhận append-only mọi thao tác INSERT/UPDATE liên quan tới semantic table (hiện tại là `memories`). Mỗi lần update (hay soft-delete), 1 record HLC mới sẽ được nối vào. Bảng sẽ phình to tỷ lệ thuận với số lượng thao tác.
- **Row Hash & Merkle Bucket Size**: Kích thước bảng `row_hash` được kiểm soát hoàn hảo vì tuân theo nguyên tắc `1 Row = 1 Hash` (Sử dụng Primary Key `table_name, row_id` kèm mệnh đề `REPLACE`). `merkle_bucket` cố định số lượng bằng `num_buckets`, hoàn toàn an toàn và miễn nhiễm với memory bloat.
- **Sync Transport (Pull Flow)**: API `fetch_delta_for_buckets` hiện đang dựa vào `row_id` sinh ra từ `row_hash` để `SELECT` trực tiếp từ bảng đích (`memories`) lấy state mới nhất. Không bê lịch sử `oplog` đi nên Data Packet rất nhẹ gọn.

## 2. Rủi Ro Thường Trực (Risks & Gaps)
- **Oplog Infinity Bloat**: Không có cơ chế cắt tỉa (Garbage Collection). `oplog` sẽ phình vô tận theo thời gian, dù giá trị của dữ liệu `oplog` thực tế chỉ cần tồn tại cho đến khi Replia chậm nhất trong Cluster sync thành công.
- **Physical Delete Orphan / Sync Stall**: Thuật toán `fetch_delta_for_buckets` đang skip những `row_id` không `SELECT` được ra data thật (vì đã bị xoá vật lý - hard delete). Việc skip này khiến Node B không bao giờ nhận được chỉ thị xoá, từ đó không thể xoá `row_hash` trên Node B. Điều này khiến hai bucket vĩnh viễn mismatch (kẹt infinite sync loop). Trái lại, Soft-Delete (`archived=1`) vẫn an toàn vì row còn tồn tại nên Node B có tải về để chạy UPDATE.

## 3. Extension Point An Toàn Nhất Để Mở Rộng
- Khái niệm **Replica Progress / Watermark** (đánh dấu replica đã kéo data đến HLC nào) chưa hề tồn tại trong Node cục bộ. Point lý tưởng và an toàn nhất để add feature này chính là bảng **`db_metadata`** hiện có của DB (Bảng dạng key/value). Chúng ta chỉ cần thiết lập keys dạng `sync_watermark_node_{node_id} = min_replicated_hlc`.
- Tiến trình Garbage Collection (GC) có thể tham chiếu min của toàn bộ các Watermark này để xoá mớ Oplog cũ phía sau an toàn.

## 4. Safe Implementation Order (Lộ Trình Hardening An Toàn)
Dưới đây là Checklist sắp xếp theo mức độ an toàn (tránh đụng chạm public API):

- [ ] **Phase 2.1: Physical Tombstone Transport**:
  - Gắn nhãn hoặc truyền struct metadata "DELETED" bên trong `fetch_delta_for_buckets` nếu row_id map với `oplog.operation == "DELETE"`. Node B `apply_remote_delta` cần có cơ chế handler bóc tác riêng để xoá nhổ vật lý + drop khỏi `row_hash`.
  
- [ ] **Phase 2.2: Replica Checkpointing**:
  - Viết method cập nhật `sync_watermark_node_{node_id}` vào `db_metadata` mỗi kết thúc quá trình sync Node.
  - Phục hồi State mỗi khi tái khởi động (để không Start Sync lại từ đầu với Full Tree Traversal nếu hệ thống có record Watermark).

- [ ] **Phase 2.3: Oplog Garbage Collection (GC Job)**:
  - Viết Background Job hoặc API gọi hàm `db.prune_oplog()`.
  - Chiến lược Prune rủi ro thấp nhất: Quét `MIN()` của tất cả Remote Watermarks. `DELETE FROM oplog WHERE version_hlc < MIN_WATERMARK`.

- [ ] **Phase 2.4: Stress Test Lifecycle**:
  - Mở cụm test chạy `for i in 10,000` tạo data -> update liên tục -> hard delete -> Prune GC -> Verify Merkle Roots match across DB.

## 5. Non-Goals của Phase Hardening
- Không implement real network server/HTTP REST ở bước này.
- Không thay thế Merkle Tree thành dạng BST phức tạp.
- Cấu trúc HLC vẫn giữ nguyên không custom logic thời gian thực vì clock sync quá phức tạp.
