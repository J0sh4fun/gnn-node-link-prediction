# Todo thành viên A

Cập nhật: **2026-10-17**. [Kế hoạch A](member_a_plan.md) · [Kế hoạch dự án](project_plan.md).

Đồng bộ todo: **2026-10-17**. [Cơ chế tự động](todo_sync.md).

PR kế hoạch và tracker: [#3 — Member A plan and progress checklist](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/3), đã merge. PR triển khai từng task được ghi riêng khi có.

## Cách theo dõi

- **To do**: chưa bắt đầu; ghi rõ blocker nếu thiếu đầu vào. **Doing**: đang làm. **Review**: đã có đầu ra và kiểm tra, còn chờ review/merge. **Done**: đã nghiệm thu và tích hợp `main`.
- Checkbox `[x]` chỉ dành cho Done. Task Review vẫn để `[ ]` dù code đã viết xong. Cập nhật cuối mỗi task và cuối tuần; thêm PR/commit và bằng chứng vào bảng nhật ký bên dưới.
- Tuần là tuần tương đối theo kế hoạch gốc. Các mục A01–A05 được ghi nhận từ artifact đã có trên `main`. Tuần 5 đã chạy lại kiểm thử ngày 2026-10-15; trạng thái tích hợp hiện tại của A06/A07 được đồng bộ trong bảng bên dưới. Không có task tuần 6 của A; B tự thực hiện toàn bộ tuần đó.
- Khi hoàn tất triển khai task: cập nhật Review → commit riêng → push → PR Ready for review vào `main`. Sau review/checks và merge mới chuyển Done. Chi tiết tại [quy trình](member_a_plan.md#quy-trình-hoàn-thành-từng-task).

## Tiến độ hiện tại

<!-- todo-sync:summary:start -->
| To do | Doing | Review | Done | Tổng |
|---:|---:|---:|---:|---:|
| 10 | 0 | 2 | 10 | 22 |

**10/22 task đã nghiệm thu và tích hợp main.** Số task không biểu thị phần trăm khối lượng.

- **A06 · Done** — [PR #4](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/4). Đã merge vào main; code/test khớp phiên bản đã nghiệm thu.
- **A07 · Done** — [PR #4](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/4). Đã merge vào main; code/test khớp phiên bản đã nghiệm thu.

Task chưa có quy tắc bằng chứng giữ trạng thái thủ công; không suy diễn từ tên file/PR.
<!-- todo-sync:summary:end -->

A08/A09 cũ đã chuyển sang B, không còn trong mẫu số. Phân công tuần 5/6 độc lập giữa hai thành viên.

Tuần 7 A10–A12 đã nghiệm thu kỹ thuật và tích hợp `main` qua
[PR #9](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/9),
nhánh `feature/a-week07-node-classification` → `main`.
[Báo cáo và lệnh tái lập](../../report/week07/member_a.md). API GitHub ngày 2026-10-17 xác nhận merge commit `7074f7d3`; đối chiếu source tuần 7 trên main và 136 passed/1 skip của suite hiện tại.

Tuần 8: GCN/GAT đã đủ đầu ra kỹ thuật; **A13/A14 Review** tại [PR #12](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/12), chờ tích hợp bản sửa dropout và artifact GAT. [Báo cáo hoàn tất](../../report/week08/completion.md). GAT test đã chạy lại theo yêu cầu người dùng, là ngoại lệ protocol một lần; không tự ghi nhận check-in/phê duyệt cá nhân.

## Checklist thực hiện

### Tuần 1–4 — Nền tảng

- [x] **A01 · Tuần 1 · Done** — Tóm tắt spectral convolution → GCN; thống nhất Problem Statement, RQ và repo. Bằng chứng: [lý thuyết GCN](../../report/GCN_Spectral_to_Formula_vi.md), [kế hoạch gốc](project_plan.md). Hoàn tất khi tài liệu giải thích đúng normalization và vai trò GCN.
- [x] **A02 · Tuần 2 · Done** — Load/preprocess Cora NC, public masks, graph vô hướng và EDA chung. Bằng chứng: [loader](../../utils/data_loader.py), [tests](../../tests/test_data_loader.py), [EDA](../../results/week02/eda_summary.json), [protocol](../protocol.md). Hoàn tất khi masks tách biệt và quy ước dữ liệu được ghi rõ.
- [x] **A03 · Tuần 3 · Done** — Baseline MLP, masked Cross-Entropy, Accuracy/Macro-F1 và phân tích RQ1. Bằng chứng: [baseline](../../train/train_mlp.py), [log final lịch sử](../../results/baseline_mlp_log.json), [phân tích](../experiments/week3_rq1_analysis_vi.md). Không chạy lại final chỉ để cập nhật tracker.
- [x] **A04 · Tuần 4 · Done** — Trainer NC, early stopping, best checkpoint, seed/logging và tracking schema. Bằng chứng: [trainer](../../train/trainer.py), [tests](../../tests/test_trainer.py), [kiến trúc](../architecture/week4_framework_vi.md), [tracking](../../results/experiment_index.csv). Hoàn tất khi train/validation tách biệt và restore checkpoint đúng.
- [x] **A05 · Tuần 4 · Done** — Utility normalization sparse, phối hợp edge/self-loop convention và bàn giao check-in 1. Bằng chứng: [graph ops](../../utils/graph_ops.py), [tests](../../tests/test_graph_ops.py), [bàn giao](../phase1_cleanup.md). Thiết kế layer tuần 4 được thay thế bằng implementation MessagePassing tuần 5; phần xác nhận thiết kế ở A07.

### Tuần 5 — A tự hoàn tất GCNLayer

- [x] **A06 · Tuần 5 · Done** — A tự cài GCNLayer, viết/chạy unit test và đối chiếu đủ sáu Acceptance Criteria. Bằng chứng: [implementation và tests](https://github.com/J0sh4fun/gnn-node-link-prediction/tree/35a5d01a47f292c66ed96c96f175fdd05329515b), [báo cáo kiểm tra](../../report/week05/progress_review.md): **66 passed**, 6/6 tiêu chí đạt. Trạng thái theo PR #4 và bằng chứng code/test; không yêu cầu B kiểm thử/review/xác nhận.
- [x] **A07 · Tuần 5 · Done** — A tự nghiệm thu, ghi API/thiết kế MessagePassing, báo cáo và chuẩn bị PR GCN vào `main`. Phụ thuộc A06. Theo dõi tích hợp qua [PR #4](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/4); không có phụ thuộc B.

### Tuần 6 — Chuyển toàn bộ sang B

Không có task tuần 6 của A. Giữ mã cũ trong bảng chuyển giao để bảo toàn lịch sử, không tái sử dụng mã cho việc khác:

| Mã cũ | Mã hiện hành của B | Nội dung |
|---|---|---|
| A08 | B06-02 | B tự viết/chạy acceptance tests cho GATLayer và tự nghiệm thu |
| A09 | B06-03 | B tự benchmark GATLayer với GATConv |

Chi tiết: [todo tuần 6 của B](member_b_week06_todo.md). A nhận GATLayer làm đầu vào cho tuần 7 sau khi B hoàn tất.

### Tuần 7–8 — Node Classification

- [x] **A10 · Tuần 7 · Done** — Đã có [GCN hai tầng](../../models/gcn.py), logits `[N, C]`, ReLU/dropout giữa tầng, mỗi tầng nhận cạnh thô. Forward/backward và trainer pass; xem [nghiệm thu tuần 7](../../report/week07/member_a.md). Đã tích hợp qua PR #9 (merge `7074f7d3`).
- [x] **A11 · Tuần 7 · Done** — Đã có [GAT hai tầng độc lập](../../models/gat.py), hidden heads concat/output heads average, ELU và feature/attention dropout. Tests mạng và trainer pass; đầu vào B06-01–B06-04 đã tích hợp main. Xem [báo cáo](../../report/week07/member_a.md); đã tích hợp qua PR #9 (merge `7074f7d3`).
- [x] **A12 · Tuần 7 · Done** — Đã chạy [runner NC](../../scripts/node_classification.py) trên Cora cùng split/preprocessing, seed 42; lưu config/environment/history/checkpoint/hash. Validation Accuracy GCN/GAT 0,776/0,800; [bảng MLP/GCN/GAT](../../results/week07/comparison.csv), [báo cáo RQ1/RQ2](../../report/week07/member_a.md). 131 passed, 1 skipped (CUDA); chưa chấm test. Weights ở `runs/week07-a/` (local, ignored), có lệnh tái tạo. Đã tích hợp qua PR #9 (merge `7074f7d3`).
- [ ] **A13 · Tuần 8 · Review** — GCN tuning 11 cấu hình đã tích hợp PR #10; đã nhận GAT config 7 từ PR #11 (8 cấu hình, seed 42, minimum val_loss). [Handoff](../experiments/week08_gat_handoff.md) ghi đầy đủ budget/preprocessing/dropout; sửa API dropout sau merge và tái lập validation khớp chính xác. Giữ nguyên 5 ablation định trước. 145 passed/1 skip CUDA; chờ tích hợp bản sửa/handoff.
- [ ] **A14 · Tuần 8 · Review** — Đủ bảng final MLP/GCN/GAT: Accuracy 56,80%/82,10%/82,20%; Macro-F1 53,87%/81,07%/81,32%. Giữ MLP/GCN lịch sử; GAT đã dựng lại checkpoint, validation khớp B rồi đánh giá lặp theo yêu cầu người dùng. [Summary/predictions](../../results/week08/final-gat/summary.json), [báo cáo mốc](../../report/week08/completion.md), 145 passed/1 skip. Ngoại lệ protocol được ghi rõ; chờ review/merge, không tự phê duyệt check-in 2.

### Tuần 9–10 — Hỗ trợ LP, ablation và lỗi

- [ ] **A15 · Tuần 9 · To do** — Hỗ trợ B dùng GCN/GAT sinh embedding, thống nhất shape/normalization, graph input và decoder interface. Phụ thuộc layer đã review. Hoàn tất khi embedding hữu hạn, gradient đi qua encoder và smoke LP chạy bằng train-only adjacency; thay đổi normalization phải được ghi cấu hình.
- [ ] **A16 · Tuần 9 · To do** — Hoàn thiện tensor adapter cho `utils.evaluate.evaluate_link_prediction` dựa trên `utils.link_metrics.lp_metrics`; giữ API NumPy của B. Test ROC-AUC/AP dùng continuous scores, kiểm tra shape/labels và không chấm held-out thật trong unit tests. Cùng B audit negative sampling, split và final retrain protocol trước run chính; LP final do B phụ trách.
- [ ] **A17 · Tuần 10 · To do** — Chạy phần ablation A trong 3–5 mục chung đã chốt ở A13: GCN có/không self-loop, 2/3/4 tầng, hidden dim; phối hợp B về heads/decoder theo danh sách cuối. Lưu validation/config/bảng, gắn RQ2/RQ4; không mở rộng phạm vi sau khi xem test.
- [ ] **A18 · Tuần 10 · To do** — Confusion matrix và lỗi theo degree trên validation: định nghĩa nhóm degree trước, ghi số mẫu và tỷ lệ lỗi mỗi nhóm; phân tích lớp dễ nhầm. Phụ thuộc A12/A14; nếu mô tả test thì chỉ dùng predictions đã lưu và không tuning lại.
- [ ] **A19 · Tuần 10 · To do** — Tổng hợp ablation, dấu hiệu over-smoothing và error analysis thành Discussion; đối chiếu LP/leakage với B, review chéo và check-in 3. Phụ thuộc A15–A18 và kết quả LP/ablation của B; nêu giới hạn nếu bằng chứng chưa đủ kết luận.

### Tuần 11–12 — Báo cáo và bàn giao

- [ ] **A20 · Tuần 11 · To do** — Viết giới thiệu, Problem Statement, GCN, NC, error analysis, kết luận; ghép phần B, đối chiếu bảng và trả lời đủ RQ1–RQ4. Phụ thuộc A14/A19 và kết quả B; đủ draft báo cáo, không điền số liệu ước lượng.
- [ ] **A21 · Tuần 11 · To do** — Hoàn thành 50% slide và demo, phối hợp ghép bản chung; chuẩn bị ví dụ input/output và giải thích hai pipeline độc lập. Phụ thuộc A20; slide draft và demo có lệnh chạy cụ thể.
- [ ] **A22 · Tuần 12 · To do** — Hoàn tất reproducibility: seed, saved splits/masks, best configs/checkpoints, versions, README với lệnh chạy từng bước; review chéo repo/báo cáo. Chạy kiểm tra phù hợp, không lặp final test để smoke; ghi nơi tải artifact bị ignore nếu cần.
- [ ] **A23 · Tuần 12 · To do** — Tập dượt cùng B, cả hai giải thích được GCN/GAT, NC/LP, leakage, RQ và giới hạn; sửa slide/báo cáo theo góp ý có bằng chứng. Phụ thuộc A20–A22.
- [ ] **A24 · Tuần 12 · To do** — Đối chiếu toàn bộ DoD Mục 13 với B: Cora đủ hai bài toán, layers acceptance, baseline, 3–5 ablation, phân tích/visualization, report/slide/code/README; ghi phê duyệt và bàn giao bản cuối đúng deadline nhóm. Phụ thuộc A22/A23 và toàn bộ phần bắt buộc của B. Citeseer/Pubmed không chặn nộp.

## Nhật ký commit, PR và bằng chứng

Mỗi task đang Doing/Review thêm một dòng khi có thay đổi. Link PR là nơi theo dõi kiểm tra, góp ý và merge. Bảng dưới chỉ chứa bằng chứng đã có; chưa tạo PR cho các task tương lai.

| Ngày | Task | Trạng thái | Commit / PR / bằng chứng | Kiểm tra và bước tiếp theo |
|---|---|---|---|---|
| 2026-10-15 | A01–A05 | Done | `main` tại `f4dea8db959c0ce6b61a0166a51359fd2b4a43bc`; các artifact link trong checklist | Đối chiếu tài liệu/code hiện có; chưa chạy lại runtime trong lần lập kế hoạch |
| 2026-10-09 | A06 | Review | `feature/gcn`: `ff878c8d1817e67faba5e8d6da64d43abd325ba9`, `35a5d01a47f292c66ed96c96f175fdd05329515b` | Bằng chứng kiểm thử lịch sử; phân công chờ B cũ đã được thay thế ngày 2026-10-15 |
| 2026-10-15 | A06/A07 | Review | [Kiểm tra tiến độ tuần 5](../../report/week05/progress_review.md), [PR #4](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/4); log và JUnit mới | 66 tests pass, đủ 6/6 tiêu chí; A chịu trách nhiệm toàn bộ, chỉ còn tích hợp PR |
| 2026-10-15 | A08/A09 | Chuyển giao | [B06-02/B06-03](member_b_week06_todo.md) | B thực hiện toàn bộ kiểm thử và benchmark tuần 6; loại khỏi số task A |
| 2026-10-16 | A10 | Review | `ffe51f35`; [GCN](../../models/gcn.py), [tests](../../tests/test_nc_networks.py) | Forward/backward, raw-edge wiring và trainer pass; chờ tích hợp PR |
| 2026-10-16 | A11 | Review | `9dae73c3`; [GAT](../../models/gat.py), [tests](../../tests/test_nc_networks.py) | Hidden concat/output average, gradients hữu hạn và trainer pass; chờ tích hợp PR |
| 2026-10-16 | A12 | Review | [Báo cáo](../../report/week07/member_a.md), [JUnit](../../results/week07/junit.xml), [results](../../results/week07/comparison.csv) | 131 passed, 1 skipped vì CPU; hai run validation Cora, không chấm test; chờ tích hợp PR |

Khi cần, thêm dòng với: `ngày | task ID | trạng thái | commit/PR/run/report | lệnh và kết quả kiểm tra; blocker; bước tiếp theo`. Không ghi Done chỉ vì đã commit hoặc push.

| 2026-10-17 | A10–A12 | Done | API GitHub PR #9: merged_at 2026-10-16T11:56:08Z, merge `7074f7d3` trên origin/main | Tuần 7 đã tích hợp; suite tuần 8 136 passed/1 skip CUDA; không chạy lại final MLP |

| 2026-10-17 | A13 | Doing | [Search/selection](../../results/week08/search/selection.json), [11 runs](../../results/week08/search.csv), [protocol](../../configs/node_classification_week08.json), [report](../../report/week08/member_a.md) | Phần GCN hoàn tất kỹ thuật; 136 passed/1 skip; cấu hình GAT và phối hợp danh sách ablation còn chờ B |

| 2026-10-17 | A14 | Doing | [Final GCN](../../results/week08/final-gcn/summary.json), predictions/logits/masks và persistent claim; [report](../../report/week08/member_a.md) | 1.000 test nodes, 82,10% Accuracy/81,07% Macro-F1; metrics kiểm chứng từ predictions; GAT/check-in 2 còn chờ B |

| 2026-10-17 | A13/A14 | Doing | [PR #10](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/10), commits `0cf2917c` (A13) / `a7a69194` (A14) | Phần GCN sẵn sàng review; còn GAT tuning/final và phối hợp B; không tự approve/merge |

| 2026-10-17 | A13 | Review | [Handoff và phân tích merge](../experiments/week08_gat_handoff.md), [config](../../results/week08/gat_handoff.json) | Nhận B config 7; sửa dropout alias, validation khớp bản gốc; 145 passed/1 skip; chờ merge bản sửa |

| 2026-10-17 | A14 | Review | [Báo cáo hoàn tất](../../report/week08/completion.md), [GAT final](../../results/week08/final-gat/summary.json), [audit](../../results/week08/completion_audit.json) | Người dùng cho phép chạy lại GAT test; validation/final khớp B, checkpoint/predictions/masks đủ; không chạy lại MLP/GCN; chờ merge, giữ ngoại lệ protocol |

| 2026-10-17 | A13/A14 | Review | [PR #12](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/12), commits `c1db6256` (A13) / `df71110d` (A14) | Đủ artifact, 145 passed/1 skip; giữ ngoại lệ GAT repeat do người dùng cho phép; chờ nghiệm thu/merge |
