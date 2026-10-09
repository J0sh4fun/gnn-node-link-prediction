# Tuần 5 — Thành viên B: kiểm chứng GCNLayer

Ngày: 2026-10-09. Nhánh: `feature/gcn`.
Implementation được review: `ff878c8d1817e67faba5e8d6da64d43abd325ba9`.
Các bổ sung kiểm thử nằm trong cùng bản bàn giao với báo cáo này.

## Trách nhiệm và nguồn bằng chứng

Theo Mục 5 và tuần 5 của [project_plan.md](../../docs/plans/project_plan.md),
B phụ trách unit test cho GCNLayer của A. Đây là lần rà soát code và chạy test
mới do Codex thực hiện theo yêu cầu triển khai phần việc của B; không phải
xác nhận rằng thành viên B đã tự đọc, chạy hoặc phê duyệt. Kết quả 57 test
trong báo cáo A là lịch sử; các kết luận dưới đây dựa trên lần chạy 66 test mới.

## Nhận xét review gửi A trong repo

Chưa phát hiện lỗi cần sửa trong GCNLayer trong phạm vi đã kiểm chứng.

- `weight` của layer là `[in_channels, out_channels]`; test sao chép
  `layer.weight.t()` sang `GCNConv.lin.weight` và sao chép cùng bias.
  Cấu hình đối chứng được ghi tường minh: `improved=False`, `cached=False`,
  `add_self_loops=True`, `normalize=True`, `flow="source_to_target"`.
- Đã đối chiếu với source GCNConv của PyG 2.7.0 được cài trong môi trường.
  Self-loop có trước degree; degree tính theo node đích; bias cộng sau tổng hợp.
  Layer dùng `propagate/message`, chuẩn hóa theo cạnh và không tạo adjacency dense.
- Graph vô hướng có thể che khuất lỗi đảo source/target. Bổ sung một diagnostic
  trực tiếp cho `propagate()` với hai cạnh đi vào một nút và kết quả tính tay.
  Đây là test hook; hợp đồng `forward()` vẫn yêu cầu graph vô hướng.
- Weighted loop trùng được utility dự án cộng lại. Test trường hợp này dùng
  kết quả tính tay, không giả định PyG có cùng quy ước xử lý loop trùng.
- So sánh PyG dùng `rtol=atol=1e-6` với float32 và `1e-12` với float64.
  Đã kiểm tra cả output, gradient input, weight và bias.

## Bổ sung kiểm thử

Bộ GCN tăng từ 20 lên **29 trường hợp**, gồm 9 trường hợp mới:

| Bổ sung | Số trường hợp | Mục đích |
|---|---:|---|
| Sai phân hữu hạn bằng `gradcheck` | 1 | Đối chiếu gradient input/weight/bias với đạo hàm số, độc lập với GCNConv |
| Đổi nhãn node và thứ tự cạnh | 1 | Output chỉ đổi theo thứ tự node, kể cả node cô lập |
| Thay trọng số trên cùng tensor cạnh | 1 | Bắt lỗi dùng lại normalization/degree cũ |
| Cạnh trùng không phải self-loop | 2 | Kiểm tra cộng trọng số, cả input có và không có edge_weight |
| Hướng truyền source → target | 1 | Kiểm tra thông điệp vào đúng node nhận |
| Trọng số NaN, +Inf, -Inf | 3 | Từ chối input không hữu hạn |

`gradcheck` dùng float64, `eps=1e-6`, `atol=1e-5`, `rtol=1e-3`.
Fixture khôi phục số thread bằng `finally` để giữ vệ sinh môi trường test.
Các ca có sẵn về bias sau tổng hợp, loop, graph rỗng, node cô lập, reset,
state_dict và input không hợp lệ cũng được rà soát và chạy lại.

## Kết quả thực thi mới

Môi trường CPU: Python **3.11.9**, PyTorch **2.7.1+cpu**, PyG **2.7.0**,
pytest **8.4.2**. Test GCN dùng seed 42 và một thread CPU.
Chạy từ root repo sau khi tạo thư mục `results/week05`:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_gcn_layer.py tests/test_graph_ops.py tests/test_trainer.py -q -s --junitxml=results/week05/member_b_junit.xml | Tee-Object -FilePath results/week05/member_b_pytest.txt
```

**66 passed in 6.29s**, exit code 0: 29 test GCN và 37 test graph utility/trainer.
Bằng chứng lưu trong [log pytest](../../results/week05/member_b_pytest.txt) và
[JUnit XML](../../results/week05/member_b_junit.xml).

Sanity check: 30 node tổng hợp, 3 lớp, các chuỗi cạnh trong từng lớp;
Adam lr=0.05, 100 epoch. Loss đo lại **1.067625 → 0.025040**, accuracy **100%**.
Assertion kiểm tra loss hữu hạn, gradient hữu hạn ở mọi epoch, loss giảm ở các
mốc cách nhau 10 epoch và loss cuối nhỏ hơn 10% loss đầu.

## Checklist Acceptance Criteria — Mục 5

| Tiêu chí | Kết quả lần chạy mới | Bằng chứng chính trong test_gcn_layer.py |
|---|---|---|
| Forward chạy được | Đạt | `test_matches_pyg_output_and_gradients` |
| Mọi tham số có gradient không None | Đạt | Test so sánh PyG, overfit; thêm `test_gradients_match_finite_differences` |
| Output và gradient không NaN/Inf | Đạt trên input hợp lệ đã thử | Test so sánh PyG và overfit; input trọng số không hữu hạn bị từ chối |
| Shape `[N, out_dim]` đúng | Đạt | Test so sánh PyG, graph rỗng/không cạnh và node cô lập |
| So sánh GCNConv cùng cấu hình | Đạt | 8 tổ hợp dtype × bias × weighted trong test so sánh PyG |
| Loss giảm ổn định trên tập nhỏ | Đạt | `test_overfits_thirty_connected_nodes` |

## Giới hạn và bàn giao

Các kết quả trên là kiểm chứng trên CPU và graph tổng hợp nhỏ. Chúng không đo
benchmark Cora, GPU, hiệu năng hoặc toàn bộ bộ test của repo. Không có đánh giá
Cora test set trong lần chạy này. Đạo hàm số kiểm tra input và tham số layer;
không khẳng định khả năng học edge_weight tại các điểm degree bằng không.

- [x] Rà soát code, bổ sung test có mục đích và chạy kiểm chứng mới.
- [x] Ghi đủ bằng chứng cho sáu Acceptance Criteria và nhận xét cho A trong repo.
- [ ] Thành viên B tự đọc/chạy lại và xác nhận hiểu kết quả.
- [ ] Nhóm phê duyệt trước khi merge vào main.

B có thể dùng lệnh trên để tự xác nhận và cập nhật hai mục cuối sau khi thực hiện.
Phần việc GAT tiếp theo thuộc tuần 6 theo project_plan.md.
