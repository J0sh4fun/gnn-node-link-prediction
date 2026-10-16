# Todo tuần 6 — B tự hoàn tất GATLayer

Cập nhật phân công: **2026-10-15**. Kiểm tra tiến độ: **2026-10-16**, theo
yêu cầu người dùng và [kế hoạch gốc](project_plan.md).

**B chịu trách nhiệm toàn bộ tuần 6**: implementation, unit test, benchmark,
tự nghiệm thu, báo cáo và PR. Không có task hoặc điều kiện chờ A kiểm thử,
review hay phê duyệt. Tuần 5 GCN thuộc toàn bộ A. Việc nhận GAT ở tuần 7 là
đầu vào của pipeline NC, không chuyển trách nhiệm tuần 6 sang A.

Đã kiểm tra implementation tại
[`f5c441e1`](https://github.com/J0sh4fun/gnn-node-link-prediction/commit/f5c441e1d274fbe8f4332a98bb04d0c62e89f1d3),
nhánh `feature/gat-layer`; [PR #6](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/6)
đang mở, không phải draft, chưa merge. Code chỉ có layer và tests, chưa có
benchmark/báo cáo nghiệm thu tuần 6 trong commit đó. Chi tiết và bằng chứng mới
tại [báo cáo tiến độ](../../report/week06/progress_review.md).

- [ ] **B06-01 · Review** — Đã có `models/gat_layer.py` kế thừa MessagePassing; tự viết forward/message, attention score và softmax theo node nhận; self-loop; concat/average heads; không dùng GATConv trong implementation. Đã kiểm tra code và test tại commit nêu trên; còn tích hợp PR #6 vào `main`.
- [ ] **B06-02 · Review** — 11 test GAT pass, kiểm tra forward/backward, finite values, shapes, GATConv parity cùng weight/config, attention sum, graph mẫu tính tay và sanity overfit. Tổng 77 test gồm hồi quy GCN/utility/trainer pass; đủ 6/6 tiêu chí layer trong phạm vi CPU đã thử. Sanity 60 epoch: loss 1.169088 → 0.346603, finite outputs/gradients, giảm tại các mốc 10 epoch. Còn tích hợp PR #6. Đây là phần chuyển từ A08 cũ.
- [ ] **B06-03 · To do** — Tự benchmark với GATConv: cùng graph, kích thước, weight/config/device/dtype; ghi warm-up, số lần lặp, cách đo, đồng bộ GPU nếu dùng, môi trường và kết quả. Không coi một ngưỡng tốc độ chưa thống nhất là pass/fail. Đây là phần chuyển từ A09 cũ.
- [ ] **B06-04 · Doing** — Đã có commit/push và PR #6 Ready for review; lần kiểm tra này lưu log/JUnit, sanity trace và báo cáo đạt 6/6 tiêu chí layer. Chưa đủ gói nghiệm thu cuối: còn benchmark B06-03, báo cáo bàn giao/API cuối của B và tích hợp PR. Trạng thái phản ánh deliverable có một phần, không xác nhận B đã trực tiếp chạy hoặc phê duyệt. Không mặc định giao A làm reviewer.

| To do | Doing (một phần đầu ra) | Review | Done | Tổng |
|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 0 | 4 |

**Layer đạt 6/6 Acceptance Criteria; toàn bộ tuần 6 chưa hoàn tất.** Done vẫn
là **0/4** vì các task chưa đủ nghiệm thu/tích hợp `main`; không diễn giải là
chưa có implementation. B06-03 chưa có bằng chứng benchmark; B06-04 còn thiếu
gói bàn giao đầy đủ. Thời gian chạy pytest không thay thế benchmark layer.

B cập nhật To do → Doing → Review → Done theo bằng chứng thực tế; Done khi đã
nghiệm thu và tích hợp `main`. Quy trình PR/checks và bảo vệ nhánh nếu có vẫn
áp dụng, không có điều kiện chờ A tham gia tuần 6.
