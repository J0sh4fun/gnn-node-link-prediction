# Todo tuần 6 — B tự hoàn tất GATLayer

Cập nhật phân công: **2026-10-15**, theo yêu cầu người dùng và [kế hoạch gốc](project_plan.md).

**B chịu trách nhiệm toàn bộ tuần 6**: implementation, unit test, benchmark,
tự nghiệm thu, báo cáo và PR. Không có task hoặc điều kiện chờ A kiểm thử,
review hay phê duyệt. Tuần 5 GCN thuộc toàn bộ A. Việc nhận GAT ở tuần 7 là
đầu vào của pipeline NC, không chuyển trách nhiệm tuần 6 sang A.

Chưa xác minh lại tiến độ GAT trong lần kiểm tra tuần 5 này; các mục dưới đây
để **To do**, không kết luận chưa có code ở mọi nhánh.

- [ ] **B06-01 · To do** — Tự cài `models/gat_layer.py` bằng MessagePassing; forward/message, attention score, softmax theo láng giềng của node nhận; self-loop; hidden heads concat, output heads average. Không dùng GATConv trong implementation.
- [ ] **B06-02 · To do** — Tự viết/chạy tests đủ Mục 5: forward, backward mọi tham số, output/gradient hữu hạn, shapes, so sánh GATConv cùng weight/cấu hình, overfit graph nhỏ loss giảm. Kiểm tra normalization attention và multi-head. Đây là phần chuyển từ A08 cũ.
- [ ] **B06-03 · To do** — Tự benchmark với GATConv: cùng graph, kích thước, weight/config/device/dtype; ghi warm-up, số lần lặp, cách đo, đồng bộ GPU nếu dùng, môi trường và kết quả. Không coi một ngưỡng tốc độ chưa thống nhất là pass/fail. Đây là phần chuyển từ A09 cũ.
- [ ] **B06-04 · To do** — Tự đối chiếu 6/6 Acceptance Criteria; lưu log, benchmark, API/hướng dẫn dùng và giới hạn; commit riêng theo task, push và tạo PR Ready for review vào `main`. Theo dõi merge riêng, không mặc định giao A làm reviewer.

Tổng: **0/4 hoàn tất đã xác minh trong lần rà soát này**. B cập nhật trạng thái
To do → Doing → Review → Done dựa trên bằng chứng thực tế; Done khi đã nghiệm
thu và tích hợp `main`. Quy trình PR/checks và bảo vệ nhánh nếu có vẫn áp dụng.
