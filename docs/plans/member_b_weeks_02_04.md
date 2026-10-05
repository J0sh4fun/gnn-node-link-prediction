# KẾ HOẠCH TRIỂN KHAI TUẦN 2–4 — THÀNH VIÊN B

> Cập nhật đường dẫn sau tích hợp: cây thư mục và quy định notebook-only bên dưới là thiết kế lịch sử của nhánh B. Mục 4 và liên kết cụ thể đã cập nhật; dùng [mục lục notebook hiện tại](../../notebooks/README.md) cho cấu trúc theo mục đích và các module Python chuẩn.

**Dự án:** Phân loại nút và dự đoán liên kết trên đồ thị trích dẫn bằng GNN  
**Phiên bản kế hoạch:** 2.1 — triển khai cá nhân của B, độc lập trước khi merge  
**Ngày cập nhật:** 28/09/2026  
**Người thực hiện:** Thành viên B, có agent hỗ trợ  
**Định dạng thực hành:** Jupyter Notebook `.ipynb`; giải thích và báo cáo bằng tiếng Việt

> Đây là đặc tả để agent triển khai repository. Cấu trúc dưới đây là cấu trúc đích; việc cập nhật kế hoạch không có nghĩa notebook, dữ liệu, kết quả hoặc repository đã được tạo. Chỉ tạo và đánh dấu hoàn thành sản phẩm sau khi thực sự triển khai, kiểm tra.

## 1. Phạm vi và kết quả cần đạt

B phụ trách GAT và Link Prediction. Tuần 2 xây dữ liệu và kiểm tra leakage; tuần 3 xây cosine baseline; tuần 4 chuẩn bị utilities, MessagePassing và training framework. Mục tiêu cuối tuần 4 là demo dữ liệu → split đã kiểm tra → baseline → framework/prototype, đồng thời giải thích được các bước.

Kế hoạch này chỉ giao việc cho B và agent trên branch làm việc của B. Branch của A chưa merge và chỉ được tham khảo về cách tổ chức thư mục. B tự tạo đủ notebook, helper, dữ liệu và kiểm chứng để chạy độc lập; không import module, tải file code hoặc yêu cầu artifact từ branch A. Việc tích hợp, review chéo và hợp nhất implementation được để cho giai đoạn sau, khi người dùng giao nhiệm vụ đó.

Các quy tắc bắt buộc:

- Cora là dữ liệu ưu tiên; Citeseer/Pubmed để sau khi hoàn tất phạm vi bắt buộc.
- Đồ thị vô hướng; lưu bằng danh sách cạnh sparse.
- Code thực hành, hàm dùng chung và kiểm thử đều nằm trong `.ipynb` trong giai đoạn này.
- Configuration dùng JSON; báo cáo và hướng dẫn dùng Markdown; dữ liệu số dùng NPZ/CSV/JSON tùy mục đích.
- Tự viết GAT hoàn chỉnh ở tuần 6. Tuần 4 mới xây utilities, prototype và hợp đồng thiết kế.
- Tuần 2–4 đánh giá baseline trên **validation**. Trường test score để trống cho đến đợt final evaluation đã chốt; không chạy dự đoán test trong vòng phát triển.
- Mọi kết quả phải gắn với split, configuration và lần chạy xác định.

## 2. Kiến trúc repository

Tham khảo branch `feature/gcn` tại commit `a61ecd7` để học cách phân nhóm `docs/`, `report/`, `notebooks/`, `results/`, `data/` và tách trách nhiệm. Phần B áp dụng yêu cầu notebook của người dùng: implementation B đặt trong `.ipynb`, không sao chép các module `.py` của A. Cấu trúc bên dưới dành cho phần B; không yêu cầu tổ chức lại branch A.

### 2.1. Cấu trúc đích

```text
gnn-citation-project/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data_protocol.json
│   ├── cosine_baseline.json
│   └── training_demo.json
├── docs/
│   ├── plans/
│   │   ├── project_plan.md
│   │   └── member_b_weeks_02_04.md
│   ├── protocol.md
│   ├── reproducibility.md
│   └── decisions.md
├── notebooks/
│   ├── README.md
│   ├── shared/
│   │   ├── runtime.ipynb
│   │   ├── graph_ops.ipynb
│   │   ├── metrics.ipynb
│   │   └── training.ipynb
│   └── member_b/
│       ├── week02/
│       │   ├── 01_cora_eda.ipynb
│       │   ├── 02_link_prediction_split.ipynb
│       │   └── 03_leakage_audit.ipynb
│       ├── week03/
│       │   ├── 01_cosine_baseline.ipynb
│       │   └── 02_baseline_analysis.ipynb
│       └── week04/
│           ├── 01_graph_utilities_and_message_passing.ipynb
│           ├── 02_training_framework_validation.ipynb
│           └── 03_gat_design_and_checkin.ipynb
├── data/
│   ├── README.md
│   └── cora/                                      # sinh khi tải dữ liệu; Git ignore
├── artifacts/
│   └── splits/
│       └── cora_lp_v1/
│           ├── edges.npz
│           └── manifest.json
├── results/
│   ├── experiment_index.csv
│   ├── week02/
│   │   ├── eda_summary.json
│   │   └── leakage_audit.json
│   ├── week03/
│   │   └── cosine_validation.json
│   └── week04/
│       └── checks.json
├── report/
│   ├── README.md
│   ├── templates/
│   │   └── weekly_report.md
│   ├── week01/
│   │   └── member_b.md
│   ├── week02/
│   │   ├── member_b.md
│   │   └── figures/
│   ├── week03/
│   │   ├── member_b.md
│   │   └── figures/
│   └── week04/
│       ├── member_b.md
│       └── figures/
└── runs/                                          # sinh khi chạy; Git ignore
    └── <run_id>/
        ├── config.json
        ├── environment.json
        ├── metrics.json
        ├── history.csv
        └── checkpoints/
```

**Cách đọc cấu trúc:** đây là toàn bộ các sản phẩm dự kiến đến cuối tuần 4. Tạo file theo tiến độ, không tạo file kết quả rỗng để đủ cây thư mục. `figures/` chỉ xuất hiện khi có hình. `runs/<run_id>/` chứa những file phù hợp từng loại chạy; baseline không huấn luyện thì không cần history/checkpoint.

Cấu trúc này chỉ mô tả sản phẩm cá nhân của B. Giữ tên các nhóm thư mục quen thuộc như `notebooks/`, `docs/`, `report/`, `results/`, `data/` để dễ đối chiếu với repo nhóm. `notebooks/shared/` là các hàm dùng chung giữa notebook của **B**, không phải code đã được A/B hợp nhất. Không tạo khu vực hoặc file đầu ra thay cho A.

### 2.2. Trách nhiệm của từng thư mục

| Vị trí | Nội dung | Người đọc cần tìm gì ở đây? |
|---|---|---|
| `README.md` | Giới thiệu, trạng thái, cách bắt đầu, liên kết các tuần | Điểm vào duy nhất của repository |
| `configs/` | Các cấu hình đã thống nhất | Thông số nào đang có hiệu lực? |
| `docs/plans/` | Kế hoạch gốc và kế hoạch chi tiết của B | Làm gì, khi nào, ai phụ trách? |
| `docs/protocol.md` | Quy ước dữ liệu, split, negative, validation/final test | Thí nghiệm phải tuân theo điều kiện nào? |
| `docs/reproducibility.md` | Môi trường và thứ tự chạy từ repo mới | Làm thế nào chạy lại? |
| `docs/decisions.md` | Quyết định, lý do, ngày thay đổi | Vì sao chọn phương án này? |
| `notebooks/shared/` | Định nghĩa hàm/class dùng lại | Logic chung được bảo trì ở đâu? |
| `notebooks/member_b/weekNN/` | Giải thích, chạy thí nghiệm, kiểm thử của B | B đã làm từng bước như thế nào? |
| `data/` | Hướng dẫn tải và bản dữ liệu local | Dữ liệu gốc đến từ đâu? |
| `artifacts/splits/` | Split cố định đã được kiểm tra | Model nào cũng dùng đúng tập cạnh nào? |
| `results/` | Kết quả nhỏ đã chọn để công bố cùng repo | Con số và trạng thái kiểm tra có bằng chứng gì? |
| `report/weekNN/` | Tổng hợp nghiên cứu/tiến độ tuần và hình | B học được gì, kết luận gì? |
| `runs/` | Toàn bộ output từng lần chạy tại local | Muốn debug hoặc lấy lại checkpoint ở đâu? |

Không dùng `docs/` làm nơi chứa bảng kết quả thực nghiệm thay cho `results/`. Báo cáo trích dẫn số liệu từ kết quả đã lưu, không duy trì thêm một bảng số liệu độc lập phải nhập tay.

### 2.3. Quy ước đặt tên và trách nhiệm bảo trì

- Tên đường dẫn viết thường, tiếng Anh, `snake_case`, không dấu và không khoảng trắng.
- Số thứ tự notebook bắt đầu lại trong từng tuần: `01_`, `02_`, `03_`. Không đánh số chung cả A/B khiến thêm notebook phải đổi tên hàng loạt.
- Báo cáo dùng `report/weekNN/member_b.md`; chủ đề nằm trong tiêu đề báo cáo để không phải đổi đường dẫn khi mở rộng nội dung.
- Các hàm chung chỉ có một định nghĩa chuẩn trong `shared/`. B thay đổi hàm dùng chung phải rà notebook gọi hàm đó.
- Kết quả chỉ dùng tên ổn định khi đã chọn lần chạy để bàn giao. Các lần chạy thử nằm trong `runs/` và có `run_id` riêng.
- Không tạo thư mục `src/`, `models/`, `utils/` hoặc `tests/` chứa Python song song với notebook trong phạm vi này. Test nằm trong notebook audit/validation xác định ở bảng mục 4.

### 2.4. File đưa lên GitHub và file giữ local

| Loại | Theo dõi bằng Git? | Quy tắc |
|---|---|---|
| Notebook, Markdown, cấu hình, requirements | Có | Nội dung hoàn chỉnh, đường dẫn tương đối |
| Split `edges.npz` và `manifest.json` | Có | Chỉ số cạnh nhỏ; lưu cố định sau audit, kiểm tra dung lượng trước commit |
| JSON/CSV kết quả đã chọn | Có | Có run ID, config và split fingerprint |
| Hình trong báo cáo | Có | Hình nhỏ, tên rõ nội dung, có chú thích |
| Cora tải về và dữ liệu cache | Không | `data/cora/` bị ignore; hướng dẫn tải được commit |
| Run history đầy đủ và checkpoint | Không | `runs/` bị ignore |
| Môi trường ảo, notebook checkpoint, cache Python | Không | Ignore theo thư mục cụ thể |
| Output notebook | Có chọn lọc | Giữ bảng/đồ thị quan trọng; bỏ log dài, traceback và dữ liệu lớn |

`.gitignore` cần có tối thiểu `data/cora/`, `runs/`, `.venv/`, `**/.ipynb_checkpoints/`, `**/__pycache__/`. Không ignore toàn bộ `data/` vì cần giữ `data/README.md`; không ignore toàn bộ `*.npz` vì split cố định cần được theo dõi.

## 3. Cách notebook dùng chung code và dữ liệu

### 3.1. Tái sử dụng code mà vẫn giữ `.ipynb`

Dùng IPython kernel. Notebook theo tuần nạp các notebook định nghĩa trong `notebooks/shared/` bằng `%run` với đường dẫn đã resolve từ repository root. `%run` hỗ trợ file `.ipynb` và thực thi các cell code của file đó [T1].

Vì vậy, notebook trong `shared/` phải tuân thủ:

1. Chỉ chứa import, định nghĩa hàm/class và Markdown giải thích. Ví dụ minh họa là Markdown hoặc hàm được gọi rõ ràng từ notebook tuần.
2. Khi nạp, không tự tải dữ liệu, tạo split, train, đặt seed, ghi file hoặc vẽ hình.
3. Mọi hàm nhận input/config/path qua tham số; không trông chờ biến `data`, `model`, `config` được tạo từ một notebook khác.
4. Không có phụ thuộc vòng. Notebook tuần nạp trực tiếp các shared notebook cần dùng; shared notebook không gọi ngược notebook tuần.
5. Không gọi `%run` một notebook thí nghiệm chỉ để lấy một hàm, vì có thể vô tình chạy lại thí nghiệm đó.

Sau tích hợp, ví dụ nạp code tương ứng là `from utils.link_graph import canonicalize, bidirectional` sau khi cài editable package. Notebook tham chiếu import nằm tại `notebooks/04_framework_validation/02_graph_ops_imports.ipynb`; không dùng `%run` để import notebook khác. Không viết đường dẫn máy cá nhân vào repo.

| Shared notebook | Nội dung chuẩn | Bắt đầu tạo |
|---|---|---|
| `runtime.ipynb` | Đọc config, seed, thu thập môi trường, ghi JSON và tạo run ID | Tuần 2 |
| `graph_ops.ipynb` | Chuẩn hóa/canonicalize cạnh, chuyển hai chiều, kiểm tra chỉ số; bổ sung loop/neighborhood tuần 4 | Tuần 2 |
| `metrics.ipynb` | Hàm evaluation LP của B, kiểm tra shape và nhãn | Tuần 3 |
| `training.ipynb` | Early stopping, checkpoint, điều phối train/validation qua hàm callback | Tuần 4 |

### 3.2. Đường dẫn và trạng thái kernel

- Cell bootstrap ngắn tìm root từ working directory và các thư mục cha, dựa vào sự tồn tại đồng thời của `configs/` và `notebooks/`. Nếu không tìm thấy, báo hướng dẫn mở repo; không đoán đường dẫn.
- Mọi đường dẫn được dựng bằng `pathlib` từ root đã tìm thấy.
- Có thể mở notebook trong JupyterLab hoặc VS Code bằng Python/IPython kernel của môi trường đã cài.
- Mỗi notebook phải chạy trong kernel mới khi các file đầu vào cần thiết đã tồn tại.
- Phụ thuộc giữa các tuần đi qua file đã lưu. Không có yêu cầu “chạy notebook trước để giữ biến trong RAM”.
- Thiếu artifact phải báo rõ đường dẫn thiếu và notebook cần chạy; không âm thầm tạo split khác.

### 3.3. Hợp đồng dữ liệu

**Dataset local:** dùng `data/cora/` làm root tải Cora. Giữ nguyên thứ tự node ID của dataset. Ghi fingerprint của feature matrix và danh sách cạnh gốc đã chuẩn hóa để phát hiện dataset thay đổi.

**Split cố định:** `artifacts/splits/cora_lp_v1/edges.npz` chứa sáu mảng `train_pos`, `val_pos`, `test_pos`, `train_neg`, `val_neg`, `test_neg`. Mỗi mảng có shape `[2, M]`, integer index, mỗi cột là một cặp canonical `u < v`. Các cặp không chứa self-loop và không trùng trong từng mảng.

**Manifest:** `manifest.json` ghi schema version, split ID, seed, tỷ lệ 85/5/10, cách làm tròn, tỷ lệ negative, số node, số cặp từng phần, phiên bản dataset/thư viện, fingerprint input, hash nội dung các mảng và notebook tạo split. Hash nội dung theo thứ tự/dtype/shape chuẩn hóa; không dựa riêng vào byte container NPZ có thể thay đổi metadata.

**Adjacency encoder:** dựng hai chiều từ `train_pos` khi phát triển. Self-loop được thêm tại bước chuẩn bị layer theo convention. Không lấy `edge_index` full dataset làm input encoder của LP trong tuning.

**Tái chạy:** nếu split `v1` đã có, nạp và kiểm tra fingerprint. Muốn kiểm chứng generator, sinh vào bộ nhớ hoặc thư mục run rồi so sánh với split đã lưu. Nếu đổi seed/protocol, tạo split ID mới và ghi quyết định; không ghi đè `v1` âm thầm.

**Handoff:** notebook baseline đọc feature từ dataset và cạnh từ split đã lưu; chỉ tiếp tục nếu audit đã pass và fingerprint còn khớp. Audit có thể đọc cả test pairs để kiểm tra giao tập, nhưng không tính test predictions/metrics.

### 3.4. Cấu hình và log

| File cấu hình | Trường bắt buộc |
|---|---|
| `data_protocol.json` | dataset, public node split, seed, undirected, val/test ratio, negative ratio, split ID, quy tắc self-loop |
| `cosine_baseline.json` | split ID, feature preprocessing, epsilon, batch/chunk size nếu cần, evaluation split = validation |
| `training_demo.json` | demo model/task, seed, device, learning rate, weight decay, max epochs, patience, monitor, mode, min_delta |

Config là nguồn thông số chuẩn; notebook in lại config hiệu lực rồi thực thi. Nếu có override, lưu cả config hiệu lực vào `runs/<run_id>/config.json`.

Mỗi lần chạy lưu môi trường và kết quả thật. `results/experiment_index.csv` chỉ liệt kê các run đã chọn để bàn giao với các cột:

`run_id, week, owner, task, model, dataset, split_id, split_hash, seed, config_path, code_revision, dirty_worktree, validation_metric, validation_value, test_value, result_path`.

`test_value` để trống tuần 2–4. Với toy test không dùng Cora, dataset là `toy`, các trường split không áp dụng ghi rõ null/rỗng. Không tạo giá trị giả để điền đủ cột.

## 4. Danh mục notebook và thứ tự chạy của B

Các đường dẫn hiện tại trong cột notebook tính từ `notebooks/`; ID W2–W4 giữ nguyên để truy vết lịch sử.

| ID | Notebook | Input | Output chính | Trách nhiệm |
|---|---|---|---|---|
| W2.1 | `01_data_exploration/02_cora_link_prediction_profile.ipynb` | Config, dataset tải qua API | EDA summary, figures tuần 2 | B |
| W2.2 | `03_link_prediction/01_edge_split.ipynb` | Dataset đã chuẩn hóa, data config | Split NPZ + manifest | B |
| W2.3 | `03_link_prediction/02_leakage_audit.ipynb` | Dataset và split đã lưu | Audit JSON, assertions | B và agent kiểm chứng |
| W3.1 | `03_link_prediction/03_cosine_baseline.ipynb` | Feature, split, audit, cosine config | Validation metrics và run metadata | B |
| W3.2 | `03_link_prediction/04_cosine_analysis.ipynb` | Run kết quả W3.1 | Bảng, figures, kiểm tra lặp lại | B |
| W4.1 | `04_framework_validation/05_graph_utilities_and_message_passing.ipynb` | Toy graphs, graph helpers | Kết quả utility/propagate tests | B |
| W4.2 | `04_framework_validation/06_training_framework_validation.ipynb` | Toy task, training config | History, checkpoint local, validation checks | B |
| W4.3 | `05_model_design/01_gat_design_and_checkin.ipynb` | Kết quả W2–W4 | Bảng shape, thiết kế GAT, tổng hợp check-in | B |

Thứ tự dữ liệu: W2.1 → W2.2 → W2.3 → W3.1 → W3.2. W4.1 và W4.2 độc lập về dữ liệu; W4.3 tổng hợp cuối. Nạp shared notebook nằm ngay trong notebook gọi, không yêu cầu người đọc tự chạy `shared/` theo thứ tự.

## 5. Chuẩn báo cáo theo tuần

### 5.1. Vị trí và liên kết

- Báo cáo tuần 1 đã có: đặt tại `report/week01/member_b.md` khi dựng repo.
- Tuần 2–4: mỗi tuần một `report/weekNN/member_b.md` và các hình tại `figures/` cùng thư mục.
- `report/README.md` có bảng tuần, chủ đề, liên kết báo cáo, trạng thái và ngày cập nhật.
- README gốc liên kết tới kế hoạch, mục lục notebook và mục lục báo cáo.
- Trong báo cáo tuần 2, notebook EDA được liên kết bằng `../../notebooks/01_data_exploration/02_cora_link_prediction_profile.ipynb`; hình bằng `figures/degree_distribution.png`; kết quả bằng `../../results/week02/eda_summary.json`.

### 5.2. Mẫu bắt buộc tại `report/templates/weekly_report.md`

1. **Thông tin:** tuần, tác giả, ngày, branch/code revision, trạng thái.
2. **Mục tiêu:** yêu cầu tuần này và tiêu chí hoàn thành.
3. **Kiến thức nghiên cứu:** giải thích vấn đề, công thức cần dùng, nguồn đọc.
4. **Cách thực hiện:** notebook nào, input gì, phương pháp nào, lý do lựa chọn.
5. **Kết quả:** bảng số liệu/hình từ run thật; dẫn run ID và file kết quả.
6. **Kiểm chứng:** các assertion/test, cách tái chạy, kết quả pass/fail.
7. **Phân tích:** kết quả có ý nghĩa gì, giới hạn nào còn tồn tại.
8. **Tự kiểm tra:** phần agent hỗ trợ, phần B đã đọc/chạy/giải thích lại và các vấn đề của B còn mở.
9. **Tiến độ:** hoàn thành/chưa hoàn thành; nguyên nhân cụ thể.
10. **Tuần tiếp theo:** đầu việc và dependency cần bàn giao.
11. **Tài liệu tham khảo:** nguồn thực sự đã sử dụng.

Báo cáo phải đủ để đọc hiểu kết quả mà không mở từng cell. Notebook cung cấp chi tiết thực hành; báo cáo liên kết tới notebook để kiểm chứng. Không đánh dấu `[x]` cho nhiệm vụ chưa có bằng chứng chạy hoặc review.

## 6. Tuần 2 — Dữ liệu Cora và split Link Prediction

### Ngày 1 — Khởi tạo cấu trúc, môi trường và điểm vào

**Agent làm:** kiểm tra repo hiện có; lập bảng chuyển đường dẫn nếu cần; tạo các thư mục/file của tuần 2; cập nhật README, requirements và hướng dẫn môi trường. Tạo `runtime.ipynb`, cấu hình dữ liệu, bootstrap notebook W2.1. Kiểm tra Python, Torch, PyG và IPython; khóa phiên bản đã chạy được thay vì đoán version.

**Đầu ra:** khung repo có README dẫn tới kế hoạch/notebook/report; W2.1 tải Cora vào đúng `data/cora/`, in phiên bản, shape và dtype.

**B cần hiểu:** vai trò từng thư mục; khác biệt feature, label, node mask và edge list.

### Ngày 2 — EDA và quy ước đồ thị

**Agent làm trong W2.1:** thống kê node/features/classes, số cột cạnh, số cặp vô hướng duy nhất, self-loop, duplicate, node cô lập, phân bố degree và lớp. Kiểm tra shape/index. Không ép số cạnh khớp paper nếu cách đếm khác.

**Shared:** định nghĩa canonicalize/loại trùng/chuyển hai chiều trong `graph_ops.ipynb`, có Markdown giải thích. W2.1 minh họa các hàm này trên vài cạnh trước khi áp dụng Cora.

**Đầu ra:** `results/week02/eda_summary.json`, hình degree/class trong `report/week02/figures/`; `docs/protocol.md` ghi rõ graph convention.

**Nghiệm thu:** phân biệt được số cặp vô hướng và số cột cạnh; chuẩn hóa không đổi node ID; không có index ngoài miền.

### Ngày 3 — Thiết kế split và hợp đồng dữ liệu cá nhân

**B + agent:** viết phần giải thích W2.2, bảng input/output đúng mục 3.3; giải thích khác biệt giữa public node masks có sẵn trong dataset và edge split do B tạo. Tỷ lệ tính trên cặp canonical không self-loop.

Chốt cách làm tròn: `n_val = floor(0.05 * M)`, `n_test = floor(0.10 * M)`, `n_train = M - n_val - n_test`. Ghi số lượng thực vào manifest. Chọn seed một lần và lưu cấu hình.

**B tự kiểm tra:** đọc public masks trực tiếp từ dataset, xác nhận node ordering và graph convention qua W2.1. Toàn bộ đầu vào phải có thể tạo bằng notebook của B.

### Ngày 4 — Tạo positive split

**Agent làm trong W2.2:** lấy unique undirected pairs; hoán vị có seed; chia ba tập; dựng train adjacency hai chiều từ train positives. Hiển thị ví dụ để phân biệt cạnh dùng supervision và cạnh message passing.

**Kiểm tra:** ba tập rời nhau, hợp bằng full canonical positives, không có self-pair, kích thước khớp quy tắc làm tròn, input gốc không bị sửa tại chỗ.

**B cần giải thích:** vì sao chia riêng `(u,v)` và `(v,u)` có thể gây leakage.

### Ngày 5 — Negative sampling và lưu split

**Agent làm:** lấy negatives không thuộc full positive set, không self-loop; canonicalize và bảo đảm unique. Lấy đủ số lượng sau canonicalization; loại trùng giữa cả ba negative split. Có thể sample một pool hợp lệ rồi chia thành ba phần để bảo đảm tách biệt. Nếu dùng sampler PyG, truyền full positive graph làm tập loại trừ và kiểm tra kết quả theo convention vô hướng.

Dùng tỷ lệ 1:1, fixed negatives cho tuần 2–4. Ghi rõ full graph chỉ phục vụ tạo tập loại trừ, không đưa vào encoder tuning.

**Đầu ra:** `edges.npz` và `manifest.json` hoàn chỉnh. Kiểm tra dung lượng, hash và khả năng load lại; split mới chỉ được bàn giao sau audit ngày 6.

### Ngày 6 — Audit độc lập

**Agent làm trong W2.3:** đọc file split từ đĩa; không dùng lại biến RAM trong W2.2. Kiểm tra disjoint positives, negatives hợp lệ/tách biệt, reverse-edge leakage, adjacency bằng đúng train pairs hai chiều và manifest khớp dữ liệu.

Thêm kiểm tra tính tái lập cùng seed và test đối chứng trên toy graph: cố ý đưa validation edge vào train adjacency, hoặc negative trùng held-out positive. Test phải xác nhận lỗi được phát hiện, bắt lỗi có chủ đích để toàn notebook vẫn kết thúc thành công; không để traceback không xử lý trong bản nộp.

**Đầu ra:** `results/week02/leakage_audit.json` ghi split hash, từng check và trạng thái; không ghi “pass” từ giá trị hard-code.

### Ngày 7 — Báo cáo và review tuần 2

Chạy W2.1–W2.3 với kernel mới, cập nhật báo cáo tuần 2 theo mẫu, README và mục lục. B tự giải thích EDA, split, negatives và audit; agent đối chiếu câu trả lời với code. Ghi và sửa các vấn đề trước khi chốt sản phẩm cá nhân.

**Tiêu chí hoàn thành tuần 2:**

- [ ] Có ba notebook W2.1–W2.3 chạy được và shared code cần thiết.
- [ ] Có split cố định, manifest, hash và audit pass.
- [ ] Có báo cáo `report/week02/member_b.md` với hình, kết quả và link hoạt động.
- [ ] Người khác biết tải dataset, load split và chạy audit từ README.
- [ ] Chạy được chỉ với các file thuộc phần B và dataset tải qua API.

## 7. Tuần 3 — Cosine baseline và phân tích

### Ngày 1 — Metric và hợp đồng evaluation

Tạo `shared/metrics.ipynb`: hàm nhận labels/scores, kiểm tra shape, finite values, đủ hai lớp trước khi tính ROC-AUC/AP. Dùng implementation metric của thư viện đã xác minh; giải thích AP không mặc định đồng nhất với diện tích PR tính bằng hình thang.

Trong W3.1, tạo ví dụ xếp hạng hoàn hảo và đảo ngược để kiểm tra metric. Ghi số positive/negative và split đang được đánh giá. Config chỉ định validation.

### Ngày 2 — Cosine trên vector nhỏ

Trong W3.1, giải thích công thức tích vô hướng chia tích chuẩn. Tính tay và kiểm tra với tensor; kiểm tra tính đối xứng, vector giống nhau, vuông góc và zero vector. Quy định zero-vector score là 0 với epsilon trong mẫu số; ghi rõ đây là convention xử lý trường hợp cosine toán học không xác định.

Tạo hàm score theo cặp, không tạo ma trận similarity toàn bộ `[N,N]`. Feature preprocessing phải khai báo trong config; không ghép nhãn vào feature.

### Ngày 3 — Chạy baseline trên validation

W3.1 nạp dataset, manifest, split và audit pass; xác nhận fingerprint. Chấm điểm validation positives/negatives cố định. Lưu config hiệu lực, môi trường và metrics vào run mới. Baseline không có encoder học hay checkpoint train.

Không sinh test scores. Train pairs có thể dùng để kiểm tra kỹ thuật nếu cần nhưng không thay thế validation trong bảng nghiệm thu.

### Ngày 4 — Phân tích kết quả

W3.2 đọc run đã chọn, hiển thị ROC-AUC/AP, số mẫu, seed, split ID, preprocessing và thời gian. Vẽ phân bố score positive/negative trên validation; phân tích các trường hợp khó nếu có dữ liệu hỗ trợ.

Lưu hình vào `report/week03/figures/`. Nếu cần danh sách dự đoán đầy đủ để phân tích, lưu trong run tương ứng; chỉ công bố bảng nhỏ cần thiết trong báo cáo.

### Ngày 5 — Kiểm tra trường hợp biên của baseline

Trong W3.1/W3.2, kiểm tra vector 0, score bằng nhau, NaN/Inf, input rỗng, index ngoài miền và số label/score không khớp. Ghi hành vi mong đợi: trường hợp hợp lệ trả kết quả hữu hạn; input sai báo lỗi rõ ràng. Dùng toy scores có đáp án biết trước để đối chiếu ROC-AUC/AP.

B giải thích được vì sao cosine là baseline không học, vì sao metric cần score liên tục và vì sao validation không thay thế final test. Viết phần phân tích này vào báo cáo tuần 3.

### Ngày 6 — Kiểm tra tái lập và đóng gói kết quả

Chạy W3.1 lại với cùng split/config vào run mới. So sánh metrics và scores trong tolerance đã ghi, không yêu cầu byte-identical trên mọi thiết bị. Lưu run được chọn sang `results/week03/cosine_validation.json`, kèm metadata đủ tái lập; cập nhật `experiment_index.csv`.

Baseline tuần 3 chỉ báo kết quả validation. Giữ trường test trống và mô tả kế hoạch final evaluation, tránh bảng cột “test” nhưng điền nhầm số validation.

### Ngày 7 — Báo cáo và nghiệm thu tuần 3

Viết `report/week03/member_b.md`: công thức, phương pháp, kết quả thật, tái lập, giới hạn và liên hệ RQ3. Không khẳng định GAT tốt hơn cosine khi chưa có GAT.

**Tiêu chí hoàn thành tuần 3:**

- [ ] W3.1/W3.2 chạy lại được, sử dụng cùng split đã audit.
- [ ] Có sanity tests cho cosine/metric và convention zero vector rõ ràng.
- [ ] Có validation ROC-AUC/AP, config và run ID.
- [ ] Test không được chấm điểm trong tuần này.
- [ ] Báo cáo tuần 3 có hình, link tới kết quả và kiểm chứng trường hợp biên.

## 8. Tuần 4 — Utilities và framework có thể dùng lại

### Ngày 1 — Thiết kế framework cá nhân của B

B và agent xác định interface: model forward, batch/input, loss, train step, validation step, metric monitor và checkpoint. Thiết kế training loop nhỏ có callback, kiểm chứng trên toy task và chuẩn bị dùng cho LP về sau. Không phải xây đầy đủ trainer cho cả hai bài toán trong tuần này.

Tạo `training_demo.json` và `shared/training.ipynb`. Với LP về sau, validation callback phải nhận adjacency train và evaluation pairs riêng. Không cho callback tự lấy graph full từ biến toàn cục.

### Ngày 2 — Training framework prototype

Trong W4.2, dùng mô hình nhỏ trên dữ liệu toy train/validation để kiểm tra optimizer, logging, best checkpoint và early stopping. Tách `train()`/`eval()`, validation không cập nhật gradient. Lưu checkpoint local trong `runs/<run_id>/checkpoints/`.

Dùng dãy validation score tự xây để kiểm tra `mode`, `patience`, `min_delta` và best epoch; load best checkpoint và kiểm tra trạng thái đã khôi phục. Có gradient hợp lệ, loss/output hữu hạn. Mọi kết quả toy ghi dataset = `toy`.

### Ngày 3 — Chuẩn hóa graph utilities

Mở rộng `shared/graph_ops.ipynb`; W4.1 minh họa và kiểm tra:

- shape/dtype/index range của cạnh;
- loại duplicate, chuyển hai chiều và tính idempotent;
- self-loop đúng một lần mỗi node;
- node cuối cô lập vẫn được giữ nhờ `num_nodes` truyền rõ;
- source = hàng 0, target = hàng 1 theo convention đã chốt;
- gom neighborhood theo node nhận.

Sau khi sửa shared helper, chạy lại audit W2.3 để phát hiện thay đổi ảnh hưởng split/adjacency. Không tái sinh split chỉ vì sửa utility.

### Ngày 4 — MessagePassing prototype

Trong W4.1, tự viết prototype cộng vector láng giềng đã chiếu trên toy graph, kế thừa `MessagePassing`. Ghi shape, chiều truyền tin và node dimension. So sánh output với vòng lặp tham chiếu tự tính trên graph nhỏ; kiểm tra gradient và hoán vị thứ tự cạnh.

B trình bày được phần `forward`, `message`, `aggregate`; phép chiếu và tổng hợp là phần code tự kiểm soát. Prototype này chưa có attention và không được gọi là GAT hoàn chỉnh.

### Ngày 5 — Thiết kế GAT và neighborhood softmax

Trong W4.3, đặt bảng shape từ `[N,F]` đến `[N,K,F_prime]`, scores `[E,K]`, aggregate `[N,K,F_prime]`, concat/average. Chốt vị trí bias/activation/dropout và interface trả attention kèm cạnh sau xử lý.

Minh họa softmax theo node nhận trên score toy; tự viết phiên bản nhỏ và so với utility tham chiếu. Kiểm tra nhóm một cạnh, nhóm nhiều cạnh, nhiều head và score lớn. Không ghép thành GAT training model tuần này.

### Ngày 6 — Rà soát độc lập phần B và tổng hợp kiểm chứng

Agent rà lại source/target, self-loop, node cô lập, softmax theo nhóm, dtype/device và gradient của utilities/prototype do B tạo. B chọn một toy graph tự tính kết quả rồi đối chiếu output, ghi rõ lỗi đã phát hiện và cách sửa trong báo cáo tuần 4.

Ghi `results/week04/checks.json` từ các assertion thực sự đã chạy của W4.1/W4.2 và phần shape/softmax trong W4.3. Phân biệt `passed`, `failed`, `not_run`; mỗi trạng thái phải gắn với phép kiểm tra và output thực tế của phần B.

### Ngày 7 — Check-in 1 và báo cáo

W4.3 tổng hợp các summary đã lưu, kiểm tra split hash thống nhất và liên kết tới notebook/báo cáo. Notebook check-in không tự chạy lại mọi notebook thí nghiệm phía sau để dựng bảng.

Demo cá nhân: EDA → split/audit → cosine validation → utility/MessagePassing → early stopping/checkpoint. Demo phải chạy được từ checkout branch B mà không lấy file từ branch A.

Viết `report/week04/member_b.md`: kết quả đạt được, lỗi đã sửa, phần pending, bàn giao tuần 5–6. Cập nhật README root với trạng thái Check-in 1 có bằng chứng.

**Tiêu chí hoàn thành tuần 4:**

- [ ] W4.1–W4.3 và shared framework có giải thích, chạy được trên môi trường ghi nhận.
- [ ] Utility và prototype được kiểm tra bằng toy graph tham chiếu.
- [ ] Framework chọn và khôi phục best checkpoint theo validation.
- [ ] Có thiết kế GAT đủ rõ cho tuần 6; chưa gán prototype là model hoàn chỉnh.
- [ ] Báo cáo tuần 4 và mục lục report/notebooks đã cập nhật.
- [ ] Check-in 1 truy được từ con số → result file → run/config/split → notebook.

## 9. Chuẩn chất lượng notebook và báo cáo

Mỗi notebook thực hành có cấu trúc:

1. Mục tiêu, chủ sở hữu, notebook ID và phạm vi.
2. Input files/config và notebook tạo ra input đó.
3. Kiến thức/công thức cần dùng, nguồn tham khảo.
4. Bootstrap, môi trường, nạp hàm chung, config hiệu lực.
5. Ví dụ nhỏ giúp hiểu phép tính trước khi dùng dữ liệu thật.
6. Thực hiện từng bước; cell ngắn, shape hiển thị ở điểm chuyển đổi quan trọng.
7. Assertions, kết quả thực tế và diễn giải.
8. Output paths, run ID và fingerprint liên quan.
9. Kết luận trong phạm vi notebook; 3–5 câu hỏi để B tự kiểm tra.

Nghiệm thu bằng `Restart Kernel and Run All`, trên điều kiện input đã công bố. Không giữ error output ngoài các lỗi được bắt có chủ đích để làm test. Không phụ thuộc thứ tự chạy thủ công bị đảo cell.

Chạy lại notebook phải không âm thầm ghi đè split cố định. Run mới đi vào thư mục riêng; chỉ cập nhật result được chọn sau khi review. Khi thay hàm shared, chạy lại các notebook phụ thuộc có liên quan.

## 10. Quy trình làm việc với agent và GitHub

### 10.1. Prompt giao việc dùng lại

> Tôi là thành viên B của dự án GNN, đang làm cá nhân trước khi merge. Branch A chỉ là tham khảo cấu trúc thư mục; không lấy implementation của A làm dependency. Hãy đọc kế hoạch này, README và quy tắc repo hiện có. Nhiệm vụ hiện tại là [ID W2.1/W2.2/...]. Trước khi sửa, xác định notebook, shared helpers, config, input/output và báo cáo tuần liên quan. Sau đó triển khai đầy đủ trong phạm vi task đã giao, giải thích bằng Markdown tiếng Việt và chạy kiểm chứng thực tế.
>
> Code phải nằm trong `.ipynb`. Dùng các định nghĩa chuẩn trong `notebooks/shared/`, không sao chép hàm sang nhiều notebook và không gọi notebook thí nghiệm để lấy hàm. Input/output theo hợp đồng của kế hoạch. Không âm thầm đổi protocol, split hay seed; không tạo kết quả giả. Nếu không chạy được do thiếu môi trường/dữ liệu, ghi chính xác phần chưa chạy và lý do.
>
> Kết thúc bằng danh sách file đã sửa, kiểm chứng đã chạy, output tạo ra, phần cần tôi review và các câu hỏi giúp tôi tự giải thích. Cập nhật báo cáo tuần với trạng thái thật. Chuẩn bị thay đổi sẵn để review trên GitHub; chỉ push/merge khi tôi đã giao việc đó. Các lựa chọn triển khai thông thường trong task có thể tự giải quyết và ghi lý do.

### 10.2. Nhịp làm việc

- Mỗi task có ID, mục tiêu, file đầu ra và acceptance criteria. Agent hoàn thành task trước khi mở rộng sang task tiếp theo chưa được giao.
- B đọc các cell giải thích, tự chạy lại và trả lời câu hỏi kiểm tra. Chỉ khi B hiểu mới đánh dấu “đã học/đã review”.
- Đầu tuần cập nhật báo cáo trạng thái `in_progress`; cuối tuần chốt kết quả và phần pending.
- Branch theo việc, ví dụ `member-b/week02-data`, `member-b/week03-baseline`, `member-b/week04-framework`.
- Commit/PR chứa notebook + config + kết quả đã chọn + báo cáo có liên quan. Không đưa toàn bộ thư mục run vào commit.
- Bản thay đổi/PR nếu có ghi cách chạy, input, kiểm chứng và kết quả. B review cùng agent. Review chéo với A hoặc merge vào branch chung là công việc sau, không phải tiêu chí chặn hoàn thành kế hoạch cá nhân này.

### 10.3. Nếu repository đã tồn tại

1. Xác nhận đúng branch cá nhân B do người dùng chỉ định, kiểm tra file và worktree trước khi sửa. Không checkout/merge/cherry-pick branch A để dựng nền cho phần B. Nếu branch B đã có công việc, tiếp tục từ đó thay vì khởi tạo lại.
2. Lập mapping đường dẫn cũ → mới; di chuyển, không để hai bản nội dung cùng được xem là bản chuẩn.
3. Cập nhật Markdown links, đường dẫn nạp notebook và output path cùng lúc.
4. Giữ dữ liệu/kết quả gốc, không tự gắn kết quả cũ với config/split mới.
5. Chạy lại các entry notebook bị ảnh hưởng. Với repo đã có commit, trình bày diff di chuyển để review.

Nếu chỉ đang tạo repository mới từ kế hoạch này, tạo thẳng cấu trúc mới, không cần tạo cấu trúc cũ rồi đổi tên.

## 11. Nghiệm thu toàn bộ tuần 2–4

| Nhóm kiểm tra | Điều kiện đạt |
|---|---|
| Cấu trúc | Mỗi loại file có vị trí duy nhất; không còn notebook rải ở root |
| Dễ học | Notebook có lý thuyết, toy example, shape, output và diễn giải |
| Dùng lại code | Helpers ở shared notebook, không sao chép implementation giữa tuần |
| Tái lập | Repo mới có thể tải Cora, load split đã lưu và chạy audit/baseline theo hướng dẫn |
| Protocol | Split vô hướng, negative hợp lệ/tách biệt, encoder adjacency đúng phạm vi |
| Evaluation | Có validation metrics; chưa dùng test scores để phát triển |
| Bằng chứng | Result file có run ID/config/split; audit/test status được sinh từ kiểm tra thật |
| Báo cáo | Tuần 1–4 có đường dẫn chuẩn; tuần 2–4 được viết theo tiến độ thực tế |
| GitHub | README dẫn đường, relative links đúng, ignore loại được cache/run lớn |
| Tính độc lập | Phần B tự chạy được; không có import, download code hoặc bước thực thi phụ thuộc branch A |

**Đầu ra cuối giai đoạn:** một phần repository của B có thể đọc trên GitHub, chạy lại theo hướng dẫn, và dùng làm nền để viết GAT ở tuần 6.

## 12. Căn cứ và tài liệu cho agent

- Kế hoạch gốc `Plan GNN.md`: đưa vào `docs/plans/project_plan.md` khi khởi tạo repo; nguồn chuẩn về vai trò và protocol của nhóm.
- Kế hoạch hiện tại: đưa vào `docs/plans/member_b_weeks_02_04.md`.
- Báo cáo nghiên cứu tuần 1 đã có: đưa vào `report/week01/member_b.md`, giữ nội dung và nguồn tham khảo.
- [T1] IPython, Built-in magic commands, `%run`: [tài liệu chính thức](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-run). Đã đối chiếu khả năng chạy `.ipynb` ngày 28/09/2026; notebook shared vẫn phải được kiểm tra với môi trường đã pin của repo.

Agent đối chiếu API PyTorch/PyG/scikit-learn với tài liệu chính thức ứng với phiên bản thực tế khi triển khai. Kế hoạch không mặc định rằng môi trường, repository hoặc kết quả chạy đã tồn tại.
