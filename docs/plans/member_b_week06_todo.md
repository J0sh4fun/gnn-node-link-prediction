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
được đồng bộ trạng thái trong bảng bên dưới. Commit kiểm tra chỉ có layer và tests, chưa có
benchmark/báo cáo nghiệm thu tuần 6 trong commit đó. Chi tiết và bằng chứng mới
tại [báo cáo tiến độ](../../report/week06/progress_review.md).

- [ ] **B06-01 · Review** — Đã có `models/gat_layer.py` kế thừa MessagePassing; tự viết forward/message, attention score và softmax theo node nhận; self-loop; concat/average heads; không dùng GATConv trong implementation. Trạng thái theo PR #6 và bằng chứng code/test tại commit nêu trên.
- [ ] **B06-02 · Review** — 11 test GAT pass, kiểm tra forward/backward, finite values, shapes, GATConv parity cùng weight/config, attention sum, graph mẫu tính tay và sanity overfit. Tổng 77 test gồm hồi quy GCN/utility/trainer pass; đủ 6/6 tiêu chí layer trong phạm vi CPU đã thử. Sanity 60 epoch: loss 1.169088 → 0.346603, finite outputs/gradients, giảm tại các mốc 10 epoch. Trạng thái tích hợp theo PR #6. Đây là phần chuyển từ A08 cũ.
- [ ] **B06-03 · To do** — Tự benchmark với GATConv: cùng graph, kích thước, weight/config/device/dtype; ghi warm-up, số lần lặp, cách đo, đồng bộ GPU nếu dùng, môi trường và kết quả. Không coi một ngưỡng tốc độ chưa thống nhất là pass/fail. Đây là phần chuyển từ A09 cũ.
- [ ] **B06-04 · Doing** — Đã có commit/push và PR #6 Ready for review; lần kiểm tra này lưu log/JUnit, sanity trace và báo cáo đạt 6/6 tiêu chí layer. Chưa đủ gói nghiệm thu cuối: còn benchmark B06-03, báo cáo bàn giao/API cuối của B và tích hợp PR. Trạng thái phản ánh deliverable có một phần, không xác nhận B đã trực tiếp chạy hoặc phê duyệt. Không mặc định giao A làm reviewer.

<!-- todo-sync:summary:start -->
| To do | Doing | Review | Done | Tổng |
|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 0 | 4 |

**0/4 task đã nghiệm thu và tích hợp main.** Số task không biểu thị phần trăm khối lượng.

- **B06-01 · Review** — [PR #6](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/6). Đã nghiệm thu; PR đang mở, chưa merge.
- **B06-02 · Review** — [PR #6](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/6). Đã nghiệm thu; PR đang mở, chưa merge.

Task chưa có quy tắc bằng chứng giữ trạng thái thủ công; không suy diễn từ tên file/PR.
<!-- todo-sync:summary:end -->

B06-03/B06-04 được cập nhật sau khi xác minh benchmark và gói bàn giao; cơ chế
đồng bộ không tự đánh dấu hoàn tất từ việc có file hoặc merge PR #6.
Thời gian chạy pytest không thay thế benchmark layer. Các báo cáo audit là
snapshot lịch sử; trạng thái hiện tại xem ở bảng tự động bên trên.

B cập nhật To do → Doing → Review → Done theo bằng chứng thực tế; Done khi đã
nghiệm thu và tích hợp `main`. Quy trình PR/checks và bảo vệ nhánh nếu có vẫn
áp dụng, không có điều kiện chờ A tham gia tuần 6.
