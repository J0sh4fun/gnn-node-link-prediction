# Kiểm tra tiến độ tuần 5 — A tự hoàn tất GCNLayer

Ngày kiểm tra: **2026-10-15**. Phạm vi: GCNLayer và sáu Acceptance Criteria tại
[Mục 5 của kế hoạch](../../docs/plans/project_plan.md).

## Kết luận

**Tuần 5 đã hoàn tất về kỹ thuật: 6/6 tiêu chí đạt, 66 tests pass.** Code nằm
trên `feature/gcn`; [PR #4](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/4)
được mở Ready for review vào `main`. Chưa ghi nhận đã merge. A chịu trách nhiệm
toàn bộ implementation, tests, tự nghiệm thu, báo cáo và PR; **không có blocker
chờ B**. Trạng thái task trong tracker vẫn Review đến khi tích hợp `main`.

Tuần 6 do B tự thực hiện GATLayer, tests, benchmark và PR; A không có task
kiểm thử, benchmark hoặc xác nhận ở tuần 6. Xem [todo tuần 6 của B](../../docs/plans/member_b_week06_todo.md).

## Bằng chứng mới

Implementation và test đã chạy:
[`35a5d01a47f292c66ed96c96f175fdd05329515b`](https://github.com/J0sh4fun/gnn-node-link-prediction/tree/35a5d01a47f292c66ed96c96f175fdd05329515b).
Các commit tiếp theo trên nhánh GCN trong lần làm việc này chỉ chỉnh tài liệu/phân công.

Lệnh thực tế từ root repo trên nhánh `feature/gcn`:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_gcn_layer.py tests/test_graph_ops.py tests/test_trainer.py -q -s --junitxml=runs/week05-a-audit/junit.xml
```

Kết quả: **66 passed in 17.15s**, exit code **0**, không failed/error/skipped;
gồm **29 test GCN** và **37 test graph utility/trainer**. Lưu bản log nguyên gốc
tại [pytest output](../../results/week05/member_a_progress_pytest.txt) và
[JUnit XML](../../results/week05/member_a_progress_junit.xml). Thời gian là của
bộ test, không phải benchmark tốc độ layer.

| Tiêu chí | Tiến độ | Kiểm tra chính |
|---|---|---|
| Forward hợp lệ | Đạt | So sánh PyG, graph nhỏ và edge cases |
| Gradient mọi tham số không None | Đạt | Output/input/weight/bias parity, gradcheck và overfit |
| Output/gradient không NaN/Inf | Đạt | Các assertions trong parity/overfit; từ chối raw weights không hữu hạn |
| Shape `[N, out_channels]` đúng | Đạt | Float32/float64, node cô lập, graph rỗng/không cạnh |
| So sánh GCNConv cùng weight/config | Đạt | 8 tổ hợp dtype × bias × weighted; tolerance 1e-6 cho float32, 1e-12 cho float64 |
| Sanity overfit loss giảm | Đạt | 30 node tổng hợp, 100 epoch: loss **1.067625 → 0.025040**, accuracy **100%**; kiểm tra giảm ở mốc 10 epoch |

Rà soát implementation xác nhận kế thừa `MessagePassing`, tự viết
`forward()`/`message()`, gọi `propagate()`, thêm self-loop trước degree và
chuẩn hóa theo cạnh qua `utils.graph_ops`, bias sau aggregation. Không dùng
GCNConv trong implementation; không dựng adjacency dense. Normalization được
tính lại mỗi forward để tránh dùng nhầm graph LP. Activation/dropout nằm ở
mạng bao ngoài, thuộc tuần 7.

## Phân công hiện hành và điều kiện kết thúc

- **A06:** implementation và acceptance tests tuần 5 đã đủ; còn merge PR #4.
- **A07:** A đã tự kiểm tra, cập nhật API/báo cáo và mở PR #4; còn tích hợp PR.
- **A08/A09 cũ:** chuyển sang B06-02/B06-03; không còn tính là task hoặc blocker của A.
- Báo cáo/log cũ mang tên B là lịch sử kiểm chứng theo phân công trước, được giữ để bảo toàn provenance; không tạo nghĩa vụ B review hay phê duyệt tuần 5. Báo cáo A trên nhánh GCN đã cập nhật và báo cáo B cũ có ghi chú hết hiệu lực phân công.

Quy trình PR/checks và bảo vệ nhánh nếu có vẫn áp dụng; không tự động merge
hoặc mặc định chỉ định người còn lại làm reviewer. Xem [PR #3 cập nhật kế hoạch](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/3)
và [tracker A](../../docs/plans/member_a_todo.md).

## Giới hạn kiểm tra

Lần chạy mới dùng CPU và graph tổng hợp, không tải Cora, không tính điểm
held-out Cora, không benchmark GPU hoặc kiểm tra toàn bộ repo. Mạng GCN hai
tầng và thí nghiệm NC thuộc tuần 7, không phải phần còn thiếu của tuần 5.
Chưa đánh giá lại tiến độ implementation GAT tuần 6 trong lần rà soát này.
Đây là kiểm chứng do Codex thực hiện theo yêu cầu người dùng, không phải
khẳng định thành viên nào đã trực tiếp chạy test hoặc bấm phê duyệt GitHub.
