# Todo tuần 6 — B tự hoàn tất GATLayer

Cập nhật phân công: **2026-10-15**. Kiểm tra tiến độ: **2026-10-16**, theo
yêu cầu người dùng và [kế hoạch gốc](project_plan.md).

Đồng bộ todo: **2026-10-16**. [Cơ chế tự động](todo_sync.md).

**B chịu trách nhiệm toàn bộ tuần 6**: implementation, unit test, benchmark,
tự nghiệm thu, báo cáo và PR. Không có task hoặc điều kiện chờ A kiểm thử,
review hay phê duyệt. Tuần 5 GCN thuộc toàn bộ A. Việc nhận GAT ở tuần 7 là
đầu vào của pipeline NC, không chuyển trách nhiệm tuần 6 sang A.

Đã kiểm tra implementation tại
[`f5c441e1`](https://github.com/J0sh4fun/gnn-node-link-prediction/commit/f5c441e1d274fbe8f4332a98bb04d0c62e89f1d3),
nhánh `feature/gat-layer`; [PR #6](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/6)
được đồng bộ trạng thái trong bảng bên dưới. Đây là commit của lần audit đầu
tuần; phần benchmark và bàn giao đã được bổ sung trong job hoàn tất tuần 6.
Xem [báo cáo nghiệm thu/bàn giao](../../report/week06/member_b.md) và
[audit lịch sử](../../report/week06/progress_review.md).

PR hoàn tất benchmark và bàn giao: [#8](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/8), nhánh `feature/week06-completion` → `main`. Đã đủ deliverable kỹ thuật, trạng thái tích hợp theo bảng bên dưới.

- [x] **B06-01 · Done** — Đã có `models/gat_layer.py` kế thừa MessagePassing; tự viết forward/message, attention score và softmax theo node nhận; self-loop; concat/average heads; không dùng GATConv trong implementation. Trạng thái theo PR #6 và bằng chứng code/test tại commit nêu trên.
- [x] **B06-02 · Done** — 11 test GAT pass, kiểm tra forward/backward, finite values, shapes, GATConv parity cùng weight/config, attention sum, graph mẫu tính tay và sanity overfit. Tổng 77 test gồm hồi quy GCN/utility/trainer pass; đủ 6/6 tiêu chí layer trong phạm vi CPU đã thử. Sanity 60 epoch: loss 1.169088 → 0.346603, finite outputs/gradients, giảm tại các mốc 10 epoch. Trạng thái tích hợp theo PR #6. Đây là phần chuyển từ A08 cũ.
- [ ] **B06-03 · Review** — Đã chạy [benchmark](../../results/week06/gat_benchmark.json) trên 3 graph tổng hợp/cấu hình, 2 mode, warm-up 10 và 50 sample/lớp/mode. Cùng graph/weight/config/device/dtype; output/gradient parity pass. Ratio median custom/PyG 0.985–1.019 trên CPU đã đo. Runner/config/raw samples và provenance đầy đủ; tích hợp theo PR #8. Không đặt ngưỡng tốc độ làm pass/fail. Đây là phần chuyển từ A09 cũ.
- [ ] **B06-04 · Review** — Đã đủ [báo cáo/API bàn giao](../../report/week06/member_b.md), 6/6 Acceptance Criteria, log/JUnit mới 101 test pass và benchmark thực tế. Code layer đã merge qua PR #6; gói bổ sung nằm trong PR #8 Ready for review. Trạng thái tích hợp theo PR #8; không có phụ thuộc A và không ghi nhận phê duyệt cá nhân chưa xảy ra.

<!-- todo-sync:summary:start -->
| To do | Doing | Review | Done | Tổng |
|---:|---:|---:|---:|---:|
| 0 | 0 | 2 | 2 | 4 |

**2/4 task đã nghiệm thu và tích hợp main.** Số task không biểu thị phần trăm khối lượng.

- **B06-01 · Done** — [PR #6](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/6). Đã merge vào main; code/test khớp phiên bản đã nghiệm thu.
- **B06-02 · Done** — [PR #6](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/6). Đã merge vào main; code/test khớp phiên bản đã nghiệm thu.
- **B06-03 · Review** — [PR #8](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/8). Đã nghiệm thu; PR đang mở, chưa merge.
- **B06-04 · Review** — [PR #8](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/8). Đã nghiệm thu; PR đang mở, chưa merge.

Task chưa có quy tắc bằng chứng giữ trạng thái thủ công; không suy diễn từ tên file/PR.
<!-- todo-sync:summary:end -->

B06-03/B06-04 đã được kiểm chứng và đăng ký bằng chứng theo commit
`f8fa506ca763d6b5ffbdf253c043564abe316390`, gồm benchmark/raw log và báo cáo
không đổi. Đồng bộ theo PR #8 chỉ chuyển Done khi đủ bằng chứng và đã merge;
không suy ra hoàn tất chỉ từ tên file hoặc việc merge PR #6.
Thời gian chạy pytest không thay thế benchmark layer. Các báo cáo audit là
snapshot lịch sử; trạng thái hiện tại xem ở bảng tự động bên trên.

B cập nhật To do → Doing → Review → Done theo bằng chứng thực tế; Done khi đã
nghiệm thu và tích hợp `main`. Quy trình PR/checks và bảo vệ nhánh nếu có vẫn
áp dụng, không có điều kiện chờ A tham gia tuần 6.
