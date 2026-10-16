# Tuần 6 — B: nghiệm thu và bàn giao GATLayer

Ngày hoàn tất kỹ thuật: **2026-10-16**. Nhánh: `feature/week06-completion`.
Căn cứ: [kế hoạch đã điều chỉnh](../../docs/plans/project_plan.md) và
[todo tuần 6](../../docs/plans/member_b_week06_todo.md).

**Đã đủ deliverable kỹ thuật tuần 6:** GATLayer tự cài, acceptance tests,
benchmark với GATConv, log/config tái lập và hướng dẫn sử dụng. Phần code layer
đã vào main qua PR #6; benchmark/bàn giao ở nhánh hoàn tất, theo dõi merge
riêng trong todo. B chịu trách nhiệm toàn bộ tuần này, không có điều kiện chờ A.
Đây là công việc do Codex thực hiện theo yêu cầu người dùng trong phạm vi của
B; không khẳng định B đã trực tiếp thao tác hoặc phê duyệt GitHub.

## Implementation và API

[GATLayer](../../models/gat_layer.py) kế thừa PyG `MessagePassing`, tự viết
`forward()` và `message()`, tính attention source/target + LeakyReLU, softmax
theo các cạnh đi vào cùng node nhận, rồi cộng message có trọng số.
Implementation không gọi GATConv hoặc tạo adjacency dense; GATConv chỉ làm
đối chứng trong tests/benchmark. `aggregate='add'` dùng base class.

```python
import torch
from models import GATLayer

x = torch.randn(4, 16)
edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
hidden = GATLayer(16, 8, heads=8, concat=True, dropout=0.0)
h = hidden(x, edges)  # [4, 64], node 3 được giữ nhờ self-loop
output = GATLayer(64, 7, heads=8, concat=False)
logits = output(torch.nn.functional.elu(h), edges)  # [4, 7]
hidden.eval()
h, (attention_edges, alpha) = hidden(x, edges, return_attention_weights=True)
```

Đoạn ghép hai tầng minh họa API; chưa phải pipeline NC Cora hoàn chỉnh tuần 7.

| Tham số / tensor | Hợp đồng |
|---|---|
| `in_channels`, `out_channels` | Kích thước input và output **mỗi head**, số nguyên dương |
| `heads`, `concat` | Mặc định 8 head, concat=True: output `[N, heads*out_channels]`; concat=False: trung bình head, output `[N, out_channels]` |
| `x` | Tensor floating `[N, in_channels]`, dtype/device khớp tham số lớp |
| `edge_index` | Tensor long `[2,E]`, source ở hàng 0, target ở hàng 1; cùng device với `x`; graph vô hướng của dự án có cả hai chiều |
| Self-loop | Mặc định True; bỏ các self-loop đầu vào rồi thêm đúng một loop/node. Cạnh khác loop không tự khử trùng |
| `dropout` | Dropout trên attention đã softmax, chỉ khi train; mặc định 0.0. Feature dropout/ELU nằm ở mạng bao ngoài |
| `negative_slope`, `bias` | Mặc định 0.2; bias cộng sau concat/average. Layer trả giá trị thô, không softmax nhãn |
| `return_attention_weights=True` | Trả `(out, (edge_index_sau_loop, alpha))`; alpha `[E_sau_loop, heads]` là attention **trước dropout** |

Ở eval/dropout=0, attention cộng theo node nhận bằng 1 cho từng head. Không
nhầm attention trước dropout với trọng số đã dropout trong train. Hệ số được
tính lại mỗi forward, không cache topology. Lưu/nạp theo `state_dict()`;
chuyển kiểu/device bằng `.double()`/`.to(device)`. Graph LP thuộc tuần 9 phải
tuân thủ adjacency train-only khi tuning; layer không tự kiểm tra leakage.

## Nghiệm thu — 6/6 Acceptance Criteria

- [x] Forward chạy trên graph mẫu.
- [x] Backward tạo gradient không None cho mọi tham số.
- [x] Output/gradient hữu hạn trong các input hợp lệ đã thử.
- [x] Shape đúng cho concat và average; giữ node cô lập.
- [x] Đối chiếu GATConv với cùng weight/config; bổ sung so sánh cả gradient trong benchmark tests và mọi cấu hình benchmark thực tế.
- [x] Sanity overfit có loss giảm: bản trace đã lưu 60 epoch, **1.169088 → 0.346603**, output/gradient hữu hạn và giảm tại các mốc 10 epoch.

[Test layer](../../tests/test_gat_layer.py) còn kiểm tra tổng attention theo
node/head, ví dụ ba node tính tay, permutation cạnh và input validation.
[Tests benchmark](../../tests/test_benchmark_gat.py) kiểm tra graph sparse
tái lập/vô hướng/không trùng, output/gradient parity cho concat/average,
đủ sample, protocol không hợp lệ và khôi phục RNG/thread của caller.

Lần chạy mới: **101 passed in 5.19s**, exit 0, không failed/error/skipped:
11 GAT + 8 benchmark + 66 GCN/graph utility/trainer + 16 todo-sync.
Chạy từ root repo, sau khi tạo thư mục log:

```powershell
New-Item -ItemType Directory -Path runs/week06-completion -Force
.venv/Scripts/python.exe -c "import torch, pytest; torch.set_num_threads(1); torch.manual_seed(42); raise SystemExit(pytest.main(['tests/test_benchmark_gat.py','tests/test_gat_layer.py','tests/test_gcn_layer.py','tests/test_graph_ops.py','tests/test_trainer.py','tests/test_sync_todos.py','-q','-s','--junitxml=runs/week06-completion/junit.xml']))"
```

Bằng chứng mới: [pytest output](../../results/week06/completion_pytest.txt),
[JUnit XML](../../results/week06/completion_junit.xml). [Sanity trace trước đó](../../results/week06/gat_sanity_observation.json)
được giữ nguyên vì code/test GAT không thay đổi. Không coi loss/accuracy toy
graph là chất lượng tổng quát của Cora.

## Benchmark tái lập

Runner: [benchmark_gat.py](../../scripts/benchmark_gat.py),
config: [gat_benchmark.json](../../configs/gat_benchmark.json),
kết quả và toàn bộ samples: [raw JSON](../../results/week06/gat_benchmark.json).
Run tại commit **f539ed90e798d3d9e140a2fc24884a0c7476cc50**; JSON lưu
SHA256 chuẩn hóa UTF-8/LF của layer/runner/config/requirements và môi trường.

```powershell
.venv/Scripts/python.exe -m scripts.benchmark_gat --config configs/gat_benchmark.json --output runs/gat-benchmark/result.json
```

Môi trường: Windows x64, Intel64 Family 6 Model 158 Stepping 13; Python
3.11.9, torch 2.7.1+cpu, PyG 2.7.0, **CPU một thread, float32, seed 42**.
Mỗi case có graph/feature chung, weight được sao chép chính xác, cùng bias,
negative_slope=0.2, self-loop=True và attention dropout=0. Trước đo, so sánh
output/input gradients/mọi parameter gradient: atol=rtol=1e-5, tất cả pass;
max absolute output error ở cả ba case đo được là 0.

Mỗi mode/case warm-up **10 lần mỗi lớp**, sau đó **50 sample mỗi lớp**, đảo
thứ tự custom/PyG trước theo từng vòng để giảm bias thứ tự. Timer
`perf_counter_ns`; nếu dùng CUDA thì đồng bộ trước/sau từng sample. Inference
là eval + no_grad. Forward/backward gồm zero_grad, forward, mean-square loss
và backward; không có optimizer step hoặc gradient theo input trong phép đo.
Raw JSON có median, p25/p75, min/max và 50 sample riêng cho từng lớp/mode.

| Graph tổng hợp / cấu hình | Mode | Custom median (ms) | PyG median (ms) | Custom/PyG |
|---|---|---:|---:|---:|
| 64 nodes, 128 pairs; 16→8×2, concat | Inference | 0.494 | 0.492 | 1.004 |
| Như trên | Forward/backward | 1.044 | 1.059 | 0.985 |
| 2708 nodes, 5278 pairs; 1433→8×8, concat | Inference | 12.649 | 12.409 | 1.019 |
| Như trên | Forward/backward | 29.334 | 29.088 | 1.008 |
| 2708 nodes, 5278 pairs; 64→7×8, average | Inference | 7.990 | 7.863 | 1.016 |
| Như trên | Forward/backward | 20.275 | 20.269 | 1.000 |

Pairs là cặp vô hướng duy nhất, được biểu diễn hai chiều trước khi thêm
self-loop: case nhỏ có 320 cạnh sau loop, hai case lớn có 13264 cạnh sau
loop. Đây là graph/feature **tổng hợp**, hai case lớn chỉ tương đương kích
thước Cora, không sử dụng topology thật, đặc trưng hay nhãn Cora.

Trên máy này thời gian hai lớp gần nhau (ratio ~0.985–1.019); không coi vài
phần trăm chênh lệch là kết luận tốc độ phổ quát hoặc có ý nghĩa thống kê.
Chưa đo GPU, float64, dropout khác 0, optimizer hoặc huấn luyện full model.
Không có ngưỡng tốc độ làm pass/fail. Thời gian pytest không thay benchmark.

## Bàn giao và trạng thái tích hợp

- [x] Implementation/test GAT có trong main qua PR #6.
- [x] Đủ sáu tiêu chí layer, có log kiểm tra mới và trace sanity.
- [x] Benchmark cùng điều kiện, config, raw timings và provenance đã lưu.
- [x] API, ví dụ, self-loop/head/dropout conventions và giới hạn được ghi rõ.
- [x] Chuẩn bị commit/push, todo và PR của chính job hoàn tất tuần 6.
- [ ] Merge PR benchmark/bàn giao vào main; đây là trạng thái tích hợp, không phải phần kỹ thuật còn thiếu.

Tuần 7 dùng riêng mạng GCN và mạng GAT; không ghép hai loại layer trong một
mạng. Training NC, tuning và final Cora chưa chạy ở tuần 6. Tất cả kiểm tra
và benchmark lần này không tải dataset hoặc đánh giá Cora held-out.
[Audit đầu tuần](progress_review.md) giữ làm snapshot lịch sử; trạng thái mới
nhất nằm trong todo, không sửa lại kết luận lịch sử của lần audit đó.
