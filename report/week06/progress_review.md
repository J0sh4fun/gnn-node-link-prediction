# Kiểm tra tiến độ tuần 6 — GATLayer do B thực hiện toàn bộ

Ngày kiểm tra: **2026-10-16**. Căn cứ: [kế hoạch tuần 6 đã điều chỉnh](../../docs/plans/project_plan.md)
và [tracker B](../../docs/plans/member_b_week06_todo.md).

## Kết luận

**GATLayer đã có implementation và đạt 6/6 Acceptance Criteria trong phạm vi
kiểm tra CPU. Tuần 6 chưa hoàn tất toàn bộ:** chưa có benchmark so với GATConv
và gói báo cáo nghiệm thu/bàn giao cuối; code chưa merge vào `main`.

B chịu trách nhiệm toàn bộ implementation, tests, benchmark, tự nghiệm thu,
báo cáo và PR. Không có task hoặc điều kiện chờ A viết test, benchmark, review
hay xác nhận. Việc A nhận GAT làm đầu vào cho tuần 7 không đổi phân công này.

| Task | Tiến độ kiểm chứng | Phần còn lại |
|---|---|---|
| B06-01 — GATLayer | Review: implementation có trên `feature/gat-layer` | Tích hợp PR #6 |
| B06-02 — Acceptance tests | Review: 11 test GAT và 66 regression tests pass; 6/6 tiêu chí đạt | Tích hợp PR #6; giữ bằng chứng theo commit |
| B06-03 — Benchmark | To do: chưa tìm thấy script/config/kết quả benchmark trong commit kiểm tra | B đo GATLayer/GATConv cùng điều kiện và lưu kết quả/môi trường |
| B06-04 — Bàn giao và PR | Doing: có commit/push, PR #6; báo cáo audit này bổ sung log/trace | B hoàn thiện benchmark và báo cáo/API bàn giao cuối, rồi tích hợp PR |

Đây là đánh giá theo artifact: **0/4 Done, 2 Review, 1 Doing, 1 To do**.
Không quy đổi số task thành phần trăm khối lượng; không coi PR đã mở là đã merge.

## Nguồn code và trạng thái GitHub

- Commit kiểm tra: [`f5c441e1d274fbe8f4332a98bb04d0c62e89f1d3`](https://github.com/J0sh4fun/gnn-node-link-prediction/commit/f5c441e1d274fbe8f4332a98bb04d0c62e89f1d3).
- Diff với `main` chỉ có `models/gat_layer.py`, `models/__init__.py`, `tests/test_gat_layer.py`; chưa có artifact benchmark hoặc báo cáo tuần 6.
- [PR #6](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/6): đang mở, Ready for review, chưa merge khi kiểm tra. PR #5 đã đóng mà không merge; không dùng nó để kết luận hoàn tất.
- Tuần 5 GCN đã tích hợp qua [PR #4](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/4); không còn là blocker của tuần 6.

## Kết quả thực thi mới

Môi trường: Python **3.11.9**, PyTorch **2.7.1+cpu**, PyG **2.7.0**. Seed 42,
một thread CPU. Từ root repo trên commit kiểm tra, sau khi tạo thư mục đầu ra:

```powershell
New-Item -ItemType Directory -Path runs/week06-audit -Force
.venv/Scripts/python.exe -c "import torch, pytest; torch.set_num_threads(1); torch.manual_seed(42); raise SystemExit(pytest.main(['tests/test_gat_layer.py','tests/test_gcn_layer.py','tests/test_graph_ops.py','tests/test_trainer.py','-q','-s','--junitxml=runs/week06-audit/junit.xml']))"
```

**77 passed in 7.05s**, exit code **0**; không failure/error/skip. Gồm **11 test
GAT** và **66 test GCN/graph utility/trainer**. Bằng chứng: [pytest log](../../results/week06/progress_pytest.txt),
[JUnit XML](../../results/week06/progress_junit.xml).

| Acceptance Criteria Mục 5 | Kết quả | Bằng chứng |
|---|---|---|
| Forward chạy được | Đạt | Test concat/average và graph mẫu tính tay |
| Backward mọi tham số có gradient | Đạt | `test_gat_layer_backward_and_finite_gradients`; quan sát toàn bộ epoch sanity |
| Output/gradient không NaN/Inf | Đạt trong phạm vi đã thử | Test finite values và trace sanity |
| Output shape đúng | Đạt | `[N, heads*out_channels]` khi concat, `[N, out_channels]` khi average; node cô lập |
| So sánh GATConv cùng weight/config | Đạt | `test_gat_layer_comparable_to_pyg_gatconv`, atol/rtol 1e-5; đối chiếu output float32, concat=True, dropout=0 |
| Sanity overfit loss giảm ổn định | Đạt | Test có sẵn và lần quan sát riêng: 60 epoch, loss 1.169088 → 0.346603, giảm tại mốc 10 epoch |

Lần quan sát riêng gọi lại chính `test_gat_layer_sanity_overfit_toy_graph`
với seed 42/một thread CPU. Dùng hook tạm thời quanh `CrossEntropyLoss.forward`
để ghi loss và xác nhận output/loss hữu hạn, quanh `Adam.step` để xác nhận tất
cả tham số có gradient hữu hạn trước mỗi bước. Không sửa implementation,
optimizer hoặc nội dung test. Lưu toàn bộ 60 loss trong [sanity trace](../../results/week06/gat_sanity_observation.json).
Các mốc epoch 1/11/21/31/41/51 là **1.169088 / 0.383057 / 0.347316 /
0.346781 / 0.346643 / 0.346623**. Không khẳng định accuracy hoặc mọi epoch
giảm đơn điệu vì đây không phải nội dung phép đo này.

## Rà soát implementation và phần cần hoàn thiện

Code kế thừa `MessagePassing`, tự viết `forward()`/`message()`; chiếu đặc
trưng và attention source/target, LeakyReLU, softmax theo node nhận,
attention dropout, concat/average heads và bias sau aggregation. Self-loop
cũ được bỏ rồi thêm đúng một loop cho mỗi node. Không gọi GATConv trong
implementation, không dựng adjacency dense. Test còn kiểm tra attention
sum bằng 1 theo node/head, phép tính tay ba node, đổi thứ tự cạnh và node cô lập.

B cần hoàn tất B06-03 bằng benchmark có graph/seed, kích thước, weight/config,
device/dtype và chế độ train/eval tương đương; warm-up, số lần lặp, cách đo
thời gian, đồng bộ nếu dùng GPU, phiên bản môi trường và bảng kết quả.
**7.05s là thời gian pytest, không phải benchmark layer.**

Sau benchmark, B hoàn thiện B06-04: tổng hợp 6/6 tiêu chí và links log,
benchmark, API/shape/self-loop/dropout, ví dụ sử dụng và giới hạn trong báo cáo
bàn giao; cập nhật PR/tracker. Quy trình PR/checks và bảo vệ nhánh nếu có vẫn
áp dụng, không mặc định giao A làm reviewer. Kiểm tra hiện tại chưa phát hiện
lỗi implementation làm fail các test đã chạy; không có nghĩa toàn bộ input,
dtype/device hoặc hiệu năng đã được kiểm chứng.

## Giới hạn và nguồn kiểm chứng

Chỉ kiểm tra các test nêu trên và code ở commit đã ghim; không chạy toàn bộ
suite, tải dữ liệu, đánh giá Cora held-out, benchmark tốc độ layer hoặc GPU.
Parity có sẵn so sánh output ở một cấu hình, chưa kiểm chứng parity gradient,
float64 hoặc mọi tổ hợp head/dropout. Mạng NC hoàn chỉnh thuộc tuần 7.

Đây là kiểm tra do Codex thực hiện theo yêu cầu người dùng; báo cáo này không
khẳng định B đã trực tiếp chạy test hoặc phê duyệt PR. Không thay đổi phân công
tuần 6 và không triển khai thêm các hạng mục còn thiếu trong lần rà soát tiến độ.
