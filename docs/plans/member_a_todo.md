# Todo thành viên A

Cập nhật: **2026-10-15**. [Kế hoạch A](member_a_plan.md) · [Kế hoạch dự án](project_plan.md).

PR kế hoạch và tracker: [#3 — Member A plan and progress checklist](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/3), nhánh `docs/member-a-plan` → `main`, Ready for review. Đây là PR tài liệu; PR triển khai từng task được ghi riêng khi có.

## Cách theo dõi

- **To do**: chưa bắt đầu; ghi rõ blocker nếu thiếu đầu vào. **Doing**: đang làm. **Review**: đã có đầu ra và kiểm tra, còn chờ review/merge. **Done**: đã nghiệm thu và tích hợp `main`.
- Checkbox `[x]` chỉ dành cho Done. Task Review vẫn để `[ ]` dù code đã viết xong. Cập nhật cuối mỗi task và cuối tuần; thêm PR/commit và bằng chứng vào bảng nhật ký bên dưới.
- Tuần là tuần tương đối theo kế hoạch gốc. Các mục A01–A05 được ghi nhận từ artifact đã có trên `main`. Tuần 5 đã chạy lại kiểm thử ngày 2026-10-15: kỹ thuật hoàn tất, tracker giữ Review để theo dõi phần chưa merge. Không có task tuần 6 của A; B tự thực hiện toàn bộ tuần đó.
- Khi hoàn tất triển khai task: cập nhật Review → commit riêng → push → PR Ready for review vào `main`. Sau review/checks và merge mới chuyển Done. Chi tiết tại [quy trình](member_a_plan.md#quy-trình-hoàn-thành-từng-task).

## Tổng quan ban đầu

| To do | Doing | Review | Done | Tổng |
|---:|---:|---:|---:|---:|
| 15 | 0 | 2 | 5 | 22 |

Tiến độ tích hợp theo số task A: **5/22 Done (~23%)**. A08/A09 cũ đã chuyển sang B, không còn trong mẫu số. Riêng tuần 5 đạt **6/6 tiêu chí kỹ thuật**, đang chờ tích hợp PR, không chờ B. Các task có kích thước khác nhau; đây không phải phần trăm khối lượng hoặc thời gian dự án. Cập nhật lại bảng này khi đổi trạng thái.

## Checklist thực hiện

### Tuần 1–4 — Nền tảng

- [x] **A01 · Tuần 1 · Done** — Tóm tắt spectral convolution → GCN; thống nhất Problem Statement, RQ và repo. Bằng chứng: [lý thuyết GCN](../../report/GCN_Spectral_to_Formula_vi.md), [kế hoạch gốc](project_plan.md). Hoàn tất khi tài liệu giải thích đúng normalization và vai trò GCN.
- [x] **A02 · Tuần 2 · Done** — Load/preprocess Cora NC, public masks, graph vô hướng và EDA chung. Bằng chứng: [loader](../../utils/data_loader.py), [tests](../../tests/test_data_loader.py), [EDA](../../results/week02/eda_summary.json), [protocol](../protocol.md). Hoàn tất khi masks tách biệt và quy ước dữ liệu được ghi rõ.
- [x] **A03 · Tuần 3 · Done** — Baseline MLP, masked Cross-Entropy, Accuracy/Macro-F1 và phân tích RQ1. Bằng chứng: [baseline](../../train/train_mlp.py), [log final lịch sử](../../results/baseline_mlp_log.json), [phân tích](../experiments/week3_rq1_analysis_vi.md). Không chạy lại final chỉ để cập nhật tracker.
- [x] **A04 · Tuần 4 · Done** — Trainer NC, early stopping, best checkpoint, seed/logging và tracking schema. Bằng chứng: [trainer](../../train/trainer.py), [tests](../../tests/test_trainer.py), [kiến trúc](../architecture/week4_framework_vi.md), [tracking](../../results/experiment_index.csv). Hoàn tất khi train/validation tách biệt và restore checkpoint đúng.
- [x] **A05 · Tuần 4 · Done** — Utility normalization sparse, phối hợp edge/self-loop convention và bàn giao check-in 1. Bằng chứng: [graph ops](../../utils/graph_ops.py), [tests](../../tests/test_graph_ops.py), [bàn giao](../phase1_cleanup.md). Thiết kế layer tuần 4 được thay thế bằng implementation MessagePassing tuần 5; phần xác nhận thiết kế ở A07.

### Tuần 5 — A tự hoàn tất GCNLayer

- [ ] **A06 · Tuần 5 · Review** — A tự cài GCNLayer, viết/chạy unit test và đối chiếu đủ sáu Acceptance Criteria. Bằng chứng: [implementation và tests](https://github.com/J0sh4fun/gnn-node-link-prediction/tree/35a5d01a47f292c66ed96c96f175fdd05329515b), [báo cáo kiểm tra mới](../../report/week05/progress_review.md): **66 passed**, 6/6 tiêu chí đạt. Kỹ thuật đã hoàn tất; còn merge code vào `main`, không yêu cầu B kiểm thử/review/xác nhận.
- [ ] **A07 · Tuần 5 · Review** — A tự nghiệm thu, ghi API/thiết kế MessagePassing, báo cáo và chuẩn bị PR GCN vào `main`. Phụ thuộc A06. Báo cáo đã cập nhật và [PR #4](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/4) đã Ready for review. Chuyển Done khi PR tích hợp; không có phụ thuộc B.

### Tuần 6 — Chuyển toàn bộ sang B

Không có task tuần 6 của A. Giữ mã cũ trong bảng chuyển giao để bảo toàn lịch sử, không tái sử dụng mã cho việc khác:

| Mã cũ | Mã hiện hành của B | Nội dung |
|---|---|---|
| A08 | B06-02 | B tự viết/chạy acceptance tests cho GATLayer và tự nghiệm thu |
| A09 | B06-03 | B tự benchmark GATLayer với GATConv |

Chi tiết: [todo tuần 6 của B](member_b_week06_todo.md). A nhận GATLayer làm đầu vào cho tuần 7 sau khi B hoàn tất.

### Tuần 7–8 — Node Classification

- [ ] **A10 · Tuần 7 · To do** — Mạng GCN hai tầng riêng, logits `[N, C]`, ReLU/dropout giữa tầng, mỗi tầng nhận cạnh thô. Phụ thuộc A06/A07. Hoàn tất khi test forward/backward và tích hợp trainer pass; đường dẫn dự kiến `models/gcn.py`.
- [ ] **A11 · Tuần 7 · To do** — Mạng GAT hai tầng riêng, không trộn GCN/GAT; shape lớp ẩn/output đúng convention head. Phụ thuộc GATLayer do B tự hoàn tất theo B06-01–B06-04. Hoàn tất khi test mạng và trainer pass; đường dẫn dự kiến `models/gat.py`.
- [ ] **A12 · Tuần 7 · To do** — Runner NC cho GCN/GAT dùng cùng split/preprocessing; lưu config, validation metrics, seed, best checkpoint; bảng so sánh sơ bộ với MLP cho RQ1/RQ2. Phụ thuộc A10/A11. Mặc định validation-only, chưa chấm test; đường dẫn runner dự kiến `scripts/node_classification.py`.
- [ ] **A13 · Tuần 8 · To do** — Tuning GCN (lr, depth, hidden dim, dropout, weight decay); nhận GAT config từ B, ghi search budget/seed và metric chọn checkpoint. Phụ thuộc A12. Chốt trước 3–5 ablation tuần 10 và cách phân tích; tất cả tuning chỉ dùng validation.
- [ ] **A14 · Tuần 8 · To do** — Chốt NC configs, restore best checkpoint, final test một lần cho từng GCN/GAT đã chốt; dùng lại MLP final lịch sử. Lưu predictions, bảng Accuracy/Macro-F1, config/versions/checkpoint, liên hệ RQ1/RQ2; check-in 2. Phụ thuộc A13 và cấu hình GAT của B; không chọn biến thể tiếp theo bằng điểm test.

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

Khi cần, thêm dòng với: `ngày | task ID | trạng thái | commit/PR/run/report | lệnh và kết quả kiểm tra; blocker; bước tiếp theo`. Không ghi Done chỉ vì đã commit hoặc push.
