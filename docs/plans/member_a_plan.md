# Kế hoạch thành viên A — GCN và Node Classification

Ngày lập: **2026-10-15**. Nguồn yêu cầu: [project_plan.md](project_plan.md), đặc biệt Mục 2–7, 9, 11–15.
Theo dõi công việc tại [todo của A](member_a_todo.md).

## Phạm vi và hiện trạng

A chịu trách nhiệm chính cho `GCNLayer` tự cài đặt và pipeline Node Classification (NC); review GAT của B; hỗ trợ embedding, đánh giá Link Prediction (LP); thực hiện một nửa phần việc chung về dữ liệu, thực nghiệm, báo cáo và slide. Cora là bắt buộc; Citeseer/Pubmed chỉ làm khi các deliverable Cora đã đủ.

Tuần trong tài liệu là tuần tương đối của kế hoạch gốc, không suy ra ngày bắt đầu hoặc deadline mới từ ngày lập. Cập nhật tiến độ dựa trên artifact và review thực tế.

| Phần việc | Bằng chứng hiện có | Cách xử lý tiếp theo |
|---|---|---|
| Tuần 1–4 | [Lý thuyết GCN](../../report/GCN_Spectral_to_Formula_vi.md), [loader](../../utils/data_loader.py), [MLP](../../models/mlp.py), [phân tích baseline](../experiments/week3_rq1_analysis_vi.md), [framework](../architecture/week4_framework_vi.md), [bàn giao nền tảng](../phase1_cleanup.md) đã có trong `main` | Tái sử dụng; không chạy lại test set MLP để kiểm tra kế hoạch |
| Tuần 5 | [Implementation và báo cáo A tại commit ff878c8d](https://github.com/J0sh4fun/gnn-node-link-prediction/tree/ff878c8d1817e67faba5e8d6da64d43abd325ba9), [kiểm chứng bổ sung tại commit 35a5d01a](https://github.com/J0sh4fun/gnn-node-link-prediction/tree/35a5d01a47f292c66ed96c96f175fdd05329515b) trên `feature/gcn` | Đã có code và bằng chứng kiểm thử; còn chờ B review trực tiếp và merge vào `main` |
| Tuần 6–12 | Chưa có bằng chứng hoàn tất trong bản kế hoạch này | Giữ To do; xác minh đầu vào trước khi chuyển Doing |

Đây là kế hoạch và đối chiếu tài liệu/code đã lưu, không phải lần chạy xác minh thực nghiệm mới. Kết quả kiểm thử lịch sử và review do Codex hỗ trợ không thay cho phê duyệt trực tiếp của B.

## Lộ trình và tiêu chí nghiệm thu

| Tuần | Task trong todo | Công việc A | Đầu ra và điều kiện nghiệm thu | Phụ thuộc / phối hợp |
|---|---|---|---|---|
| 1 | A01 | Tóm tắt spectral convolution → GCN; thống nhất Problem Statement, RQ1–RQ4 | Giải thích được self-loop, symmetric normalization, hai pipeline độc lập; có đề cương và tài liệu GCN | B phụ trách lý thuyết GAT |
| 2 | A02 | Load Cora, chuẩn hóa đặc trưng, EDA và public masks cho NC | Loader có test; masks 140/500/1000 tách biệt; graph vô hướng; README ghi convention; EDA lưu số nút/cạnh/nhãn/đặc trưng | Chia đều phần EDA chung; B giữ pipeline split LP |
| 3 | A03 | Baseline MLP chỉ dùng đặc trưng; Accuracy và Macro-F1 | Có config, log, checkpoint chọn bằng validation và bảng baseline; bảo toàn kết quả final lịch sử | RQ1; phối hợp API đánh giá với B |
| 4 | A04, A05 | Trainer tái sử dụng, seed/logging/checkpoint; utility chuẩn hóa sparse; review chéo | Train loss chỉ dùng train mask; early stopping theo validation; khôi phục best checkpoint; schema tracking đủ trường; demo check-in 1 | Hợp nhất self-loop/edge conventions với B |
| 5 | A06, A07 | Bàn giao GCNLayer, xử lý góp ý review và tích hợp `main` | Đạt đủ 6 Acceptance Criteria bên dưới; B tự xác nhận review; PR code merge được | Implementation đã có trên `feature/gcn`; B phụ trách kiểm thử GCN |
| 6 | A08, A09 | Review GATLayer của B; viết test acceptance và benchmark tốc độ | Cùng weight/cấu hình với GATConv; kiểm tra output/gradient, concat/average head, attention theo node nhận, self-loop; báo cáo thời gian cùng graph/device/dtype | Chờ GAT của B; `main` hiện chưa có GATLayer hoàn chỉnh |
| 7 | A10–A12 | Ghép riêng mạng GCN hai tầng và GAT hai tầng; chạy NC trên Cora | Hai mạng độc lập, logits `[N, C]`; dùng trainer hiện có; lưu validation/config/checkpoint; so sánh sơ bộ MLP/GCN/GAT phục vụ RQ1/RQ2 | GCN/GAT qua acceptance; B hỗ trợ debug |
| 8 | A13, A14 | Tuning GCN; nhận cấu hình GAT đã tuning từ B; chốt final NC | Search space/budget/seed được ghi trước; chọn bằng validation; final test một lần cho mỗi mô hình/cấu hình đã chốt; đủ bảng Accuracy/Macro-F1 | GAT tuning do B chính; check-in 2 |
| 9 | A15, A16 | Hỗ trợ encoder embedding và hoàn thiện API ROC-AUC/AP dùng chung | Hợp đồng embedding nhất quán, mặc định không thêm L2-normalization khi chưa có quyết định; adapter tương thích `utils.link_metrics.lp_metrics`; audit không leakage trước run chính | B sở hữu train LP, decoder và `train/retrain_final.py` |
| 10 | A17–A19 | Chia đều 3–5 ablation đã chốt; A chính GCN self-loop/depth/hidden dim; confusion matrix và lỗi theo degree | Mỗi ablation gắn RQ, có config/log/bảng; dùng validation; khảo sát over-smoothing thay vì chỉ kết luận từ accuracy; phân tích lỗi có số mẫu mỗi nhóm | B chính GAT heads/decoder; check-in 3 |
| 11 | A20, A21 | Viết các phần của A; ghép báo cáo, chia đều slide và chuẩn bị demo | Giới thiệu, Problem Statement, GCN, NC, error analysis, kết luận; Discussion trả lời từng RQ; draft đầy đủ và slide/demo chạy được | B viết GAT/LP/leakage/attention; review chéo |
| 12 | A22–A24 | Rà soát code/báo cáo, reproducibility, tập dượt và bàn giao | Lệnh chạy lại rõ ràng, versions/config/splits/checkpoints đủ; cả hai trình bày được cả hai bài toán; đối chiếu DoD Mục 13 trước nộp | Hai thành viên cùng duyệt bản cuối |

## Các ràng buộc kỹ thuật

- GCN dùng `torch_geometric.nn.MessagePassing`, tự viết `forward()` và `message()`, chuẩn hóa theo cạnh sparse; thêm self-loop trước khi tính degree. Không tạo adjacency dense N×N. `GCNConv`/`GATConv` chỉ dùng làm đối chứng kiểm thử.
- Thiết kế tuần 4 về `nn.Module + sparse_spmm` là lịch sử. Implementation GCN tuần 5 kế thừa trực tiếp `MessagePassing` và tái sử dụng `utils.graph_ops`; không tạo thêm base class chỉ để khớp tên trong kế hoạch gốc. B review và ghi nhận quyết định này trước merge code.
- GCNLayer nhận cạnh thô; activation/dropout thuộc mạng bao ngoài. Không đưa hệ số đã chuẩn hóa vào layer lần nữa. Mạng GCN và mạng GAT huấn luyện riêng.
- Đồ thị vô hướng nhất quán cho NC và LP. NC transductive được dùng toàn bộ topology/đặc trưng, nhưng supervised loss chỉ thấy nhãn train; tuyệt đối không dùng điểm test để chọn mô hình.
- LP tuning chỉ message passing trên train edges; negative sampling loại trừ toàn bộ positive gốc, không trùng giữa splits, tỷ lệ 1:1. Giữ nguyên saved split. Final LP do B retrain từ đầu trên train+val edges với cấu hình và lịch huấn luyện đã chốt, rồi đánh giá test một lần theo Mục 3.3.
- Acceptance cho layer gồm: forward hợp lệ; mọi tham số có gradient; output/gradient không NaN/Inf; shape đúng; đối chiếu PyG cùng trọng số/cấu hình với tolerance ghi rõ; sanity overfit graph nhỏ có loss giảm. Lưu log và lệnh kiểm tra trong báo cáo tuần.

## Protocol thực nghiệm và xử lý mốc thời gian

Tuần 7 chỉ báo cáo **validation** cho GCN/GAT để còn tuning tuần 8. Tuần 8 mới điền bảng **final test** sau khi chốt cấu hình, khôi phục checkpoint tốt nhất theo validation. Baseline MLP đã có final lịch sử: dùng lại kết quả đó, không tuning tiếp dựa trên điểm test đã biết.

Ablation tuần 10 được chốt trước khi chạy final NC tuần 8 và chỉ so sánh bằng validation; không dùng kết quả test tuần 8 để chọn biến thể mới. Error analysis tuần 10 dùng validation cho các quyết định phát triển. Nếu báo cáo mô tả lỗi trên test, dùng predictions đã lưu từ lần final, không đánh giá lại hoặc quay lại tuning. Final evaluator cần lưu predictions để phục vụ bước này.

Mỗi run lưu tối thiểu `model`, `dataset`, `seed`, `learning_rate`, `hidden_dim`, `num_layers`, `dropout`, `num_heads` (GAT), `decoder` (LP), `validation_score`, `test_score` (chỉ final). Ghi thêm weight decay, metric chọn checkpoint, best epoch, versions, split ID và commit để tái lập. Run tuning để trống `test_score`.

Metric NC: Accuracy và Macro-F1; LP: ROC-AUC và AP. Benchmark tham khảo không phải điều kiện pass/fail. Không đổi preprocessing/split giữa các mô hình mà không ghi rõ.

## Quy trình hoàn thành từng task

1. Chọn một task đủ phụ thuộc, chuyển **Doing** trong [todo](member_a_todo.md); dùng nhánh riêng từ `origin/main` đã cập nhật, đặt tên `feature/a-<id>-<mo-ta>` hoặc `docs/a-<id>-<mo-ta>`.
2. Hoàn tất đầu ra và tiêu chí nghiệm thu. Chạy kiểm tra phù hợp phạm vi; tài liệu kiểm tra diff/link, code chạy test liên quan. Không chạy lại final test Cora để kiểm tra sửa tài liệu hoặc refactor.
3. Cập nhật task thành **Review**, thêm bằng chứng, lệnh/kết quả kiểm tra và mục còn chờ; chỉ stage file của task, giữ nguyên thay đổi ngoài phạm vi.
4. Tạo commit cho task, ví dụ `feat(A10): add two-layer GCN network`, rồi push nhánh lên GitHub ngay khi task đủ điều kiện review. Một task lớn có thể có nhiều commit; không gom nhiều task độc lập vào một commit.
5. Tạo PR trạng thái **Ready for review**, base `main`. Mô tả gồm task ID, thay đổi, tiêu chí đã đạt, kiểm tra, phụ thuộc và giới hạn. Nếu nhiều task liên quan ở cùng PR, mỗi task vẫn có commit riêng và đủ bằng chứng.
6. Sửa góp ý bằng commit mới và push vào cùng PR. Ghi URL PR trong todo sau khi có URL; không tạo commit chỉ để tự ghi hash của chính nó.
7. Chuyển **Done** và đánh dấu checkbox khi review được chấp thuận, checks bắt buộc pass, không conflict và PR đã merge vào `main`; lưu PR/merge commit trong lần cập nhật tiếp theo. Mở PR không đồng nghĩa được phê duyệt hoặc tự động merge.

Đầu/cuối mỗi tuần rà soát todo với B, chia đều phần việc chung và ghi blockers cụ thể. Không coi review do công cụ hỗ trợ là phê duyệt của thành viên B.

## Mốc kiểm tra và thứ tự ưu tiên

| Mốc | Điều kiện đối với A |
|---|---|
| Cuối tuần 4 | A01–A05: dữ liệu, baseline, framework và convention đã bàn giao |
| Cuối tuần 8 | A06–A14: layer qua acceptance, review/merge code, NC final đúng protocol |
| Cuối tuần 10 | A15–A19: hỗ trợ LP không leakage, 3–5 ablation chung và error analysis đủ bằng chứng |
| Cuối tuần 12 | A20–A24: báo cáo, slide, demo, reproducibility và DoD toàn dự án được đối chiếu |

Ưu tiên kế tiếp: review/merge GCN tuần 5 → test/review GAT tuần 6 → hai mạng NC → tuning/final → hỗ trợ LP/ablation → báo cáo và bàn giao. Khi B chưa bàn giao GAT, A có thể chuẩn bị mạng GCN, cấu hình và validation runner; phần GAT giữ blocker rõ ràng. Khi thiếu thời gian, bỏ mở rộng Citeseer/Pubmed trước, giữ deliverable bắt buộc Cora và 3–5 ablation đã chốt.
