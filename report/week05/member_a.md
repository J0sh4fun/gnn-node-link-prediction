# Tuần 5 — Thành viên A: GCNLayer

Ngày: 2026-10-09. Nhánh: `feature/gcn`.

## Cập nhật phân công và nghiệm thu — 2026-10-15

Theo điều chỉnh của người dùng, **A chịu trách nhiệm toàn bộ tuần 5**: cài đặt,
kiểm thử, tự kiểm tra sáu Acceptance Criteria, báo cáo và chuẩn bị PR.
Không có task hoặc điều kiện chờ B ở tuần này. Tuần 6 do B tự thực hiện toàn bộ
GATLayer, kiểm thử và benchmark; A không có task tuần 6.

Đã chạy lại trên implementation `35a5d01a47f292c66ed96c96f175fdd05329515b`:
`pytest tests/test_gcn_layer.py tests/test_graph_ops.py tests/test_trainer.py -q -s`.
Kết quả mới: **66 passed in 17.15s**, exit code 0; 29 test GCN và 37 test
utility/trainer. Sanity loss **1.067625 → 0.025040**, accuracy **100%** trên
30 node tổng hợp. Tuần 5 đạt **6/6 tiêu chí kỹ thuật**; còn bước tích hợp PR
vào `main`, không phụ thuộc B. Log gốc của lần chạy này nằm ở
`runs/week05-a-audit/`; bản bằng chứng được lưu cùng PR cập nhật kế hoạch.

## Kết quả bàn giao

Đã triển khai [GCNLayer](../../models/gcn_layer.py) bằng
`torch_geometric.nn.MessagePassing`, dùng `propagate()`, tự viết
`forward()` và `message()`. Lớp dùng tổng hợp cộng có sẵn của base class,
không gọi `GCNConv` trong implementation và không tạo adjacency dense.

Repo tuần 4 có `ProjectedSum` làm primitive thử nghiệm; chưa có lớp mang tên
`MessagePassingLayer`. GCNLayer kế thừa trực tiếp base class PyG và tái sử dụng
`utils.graph_ops`. Quyết định này thực hiện yêu cầu tuần 5 hiện tại, thay cho
đề xuất `nn.Module + sparse_spmm` trong lộ trình cũ ở tài liệu tuần 4.
Mạng GCN hai tầng và thực nghiệm Cora chính thức thuộc tuần 7.

## Giao diện GCNLayer

```python
from models import GCNLayer

layer = GCNLayer(in_channels=1433, out_channels=16, bias=True)
h = layer(x, edge_index)  # [num_nodes, 16]
# Hoặc layer(x, edge_index, edge_weight) với trọng số adjacency thô.
```

- `x`: số thực, shape `[N, in_channels]`, cùng dtype/device với tham số.
- `edge_index`: `torch.long`, shape `[2, E]`, source → target, cùng device.
  Pipeline phải cung cấp đồ thị vô hướng với cả hai chiều.
- `edge_weight`: tùy chọn, shape `[E]`, hữu hạn, không âm và cùng
  dtype/device với `x`. Không truyền hệ số đã chuẩn hóa.
- Self-loop được bổ sung trước khi tính degree. Loop không trọng số trùng
  được thu về một; weighted duplicates được cộng; loop có trọng số sẵn được giữ.
- `weight` có shape `[in_channels, out_channels]`, khởi tạo Xavier uniform.
  Bias bằng không lúc khởi tạo và được cộng sau tổng hợp.
- Activation/dropout nằm trong mạng bao ngoài. Layer trả giá trị thô.
- Không cache normalization, để đồ thị thay đổi không dùng hệ số cũ.
- Chuyển dtype/device bằng `.double()`, `.to(...)`; lưu và nạp qua
  `state_dict()`. Trainer tuần 4 có thể gọi `model(x, edge_index)`.

## Checklist Acceptance Criteria — Mục 5

- [x] Forward trên graph nhỏ chạy được.
- [x] Backward sinh gradient không None cho tất cả tham số học được.
- [x] Output và gradient hữu hạn.
- [x] Output đúng shape `[N, out_channels]`, kể cả nút cô lập và graph rỗng.
- [x] Đối chiếu output và gradient với GCNConv: cùng weight/bias, graph và cấu
  hình; bao gồm float32/float64, có/không bias, có/không trọng số cạnh.
  Dùng rtol/atol 1e-6 cho float32 và 1e-12 cho float64.
- [x] Sanity overfit 30 nút: loss giảm qua các mốc mỗi 10 epoch.

## Bằng chứng kiểm thử

Môi trường: Python trong `.venv`, PyTorch `2.7.1+cpu`, PyG `2.7.0`.
Chạy từ root repo:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_gcn_layer.py tests/test_graph_ops.py tests/test_trainer.py -q -s
```

Kết quả: **57 passed**. Bộ GCN gồm 20 trường hợp, trong đó có so sánh số học,
gradient theo input/weight/bias, tính tay degree sau self-loop, graph không
cạnh, loop trùng, degree bằng không, thay đổi topology, reset/state_dict và
từ chối input không hợp lệ. GCNConv chỉ được dùng làm đối chứng trong test.

Sanity check dùng 30 nút tổng hợp, 3 lớp, cạnh nối thành chuỗi trong từng lớp,
seed 42, Adam lr=0.05, 100 epoch. Loss từ **1.067625 → 0.025040**;
accuracy trên tập thử **100%**. Đây là kiểm tra khả năng học, không phải
kết quả Cora hoặc bằng chứng chất lượng tổng quát. Chưa đánh giá Cora test set.

## Nghiệm thu và bước tiếp theo

- [x] Implementation, kiểm thử và tài liệu API đầy đủ.
- [x] Kiểm tra lại sáu Acceptance Criteria bằng lần chạy mới ngày 2026-10-15.
- [x] Ghi rõ quyết định kế thừa MessagePassing và quy ước self-loop.
- [ ] Tích hợp code qua PR vào main theo quy trình GitHub của repo.

[Báo cáo kiểm chứng cũ](member_b.md) và log mang tên member_b được giữ làm
lịch sử bằng chứng; không còn là phân công hoặc yêu cầu B xác nhận tuần 5.
Kiểm chứng do Codex thực hiện theo yêu cầu người dùng, không khẳng định một
thành viên đã tự chạy hoặc phê duyệt. Tuần 7 mới ghép mạng hai tầng và chạy
NC Cora; việc đó không chặn hoàn tất layer tuần 5.
