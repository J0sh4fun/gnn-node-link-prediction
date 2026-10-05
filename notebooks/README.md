# Notebook theo mục đích

Các notebook Phase 1 được tổ chức theo nội dung. Số thứ tự là **thứ tự đọc gợi ý**,
không phải yêu cầu chạy tuần tự tất cả notebook. ID W2–W4 và tác giả trong phần
mở đầu được giữ để đối chiếu báo cáo lịch sử; bảng dưới ghi nguồn gốc từng file.

## Cấu trúc

```text
notebooks/
├── README.md
├── 01_data_exploration/
│   ├── 01_cora_structure_and_node_labels.ipynb
│   └── 02_cora_link_prediction_profile.ipynb
├── 03_link_prediction/
│   ├── 01_edge_split.ipynb
│   ├── 02_leakage_audit.ipynb
│   ├── 03_cosine_baseline.ipynb
│   └── 04_cosine_analysis.ipynb
├── 04_framework_validation/
│   ├── 01_runtime_imports.ipynb
│   ├── 02_graph_ops_imports.ipynb
│   ├── 03_metrics_imports.ipynb
│   ├── 04_training_imports.ipynb
│   ├── 05_graph_utilities_and_message_passing.ipynb
│   └── 06_training_framework_validation.ipynb
└── 05_model_design/
    └── 01_gat_design_and_checkin.ipynb
```

Không tạo `02_node_classification/` vì hiện chưa có notebook MLP hay phân tích
node classification riêng. Pipeline này nằm trong
[train/train_mlp.py](../train/train_mlp.py), [models/mlp.py](../models/mlp.py) và
[các tài liệu RQ1](../docs/experiments/week3_rq1_analysis_vi.md).
Notebook EDA đầu tiên phân tích nhãn nút nhưng không huấn luyện mô hình.

## Danh mục và hợp đồng dữ liệu

Đường dẫn input/output trong bảng tính từ **repository root**. `runs/<id>/`
là output sinh khi chạy, không phải file được bảo đảm có trong checkout.

| Notebook | Mục đích | Task | Nguồn gốc | Input cần có | Output |
|---|---|---|---|---|---|
| [01_data_exploration/01_cora_structure_and_node_labels.ipynb](01_data_exploration/01_cora_structure_and_node_labels.ipynb) | So sánh trước/sau ToUndirected, histogram/log–log, hubs, nhãn toàn graph và train mask; giải thích chuẩn hóa GCN | NC / kiến thức graph chung | A, Tuần 2 | Cora qua Planetoid, root `data/` | Biểu đồ và thống kê inline; có thể tạo cache nếu thiếu; không publish summary |
| [01_data_exploration/02_cora_link_prediction_profile.ipynb](01_data_exploration/02_cora_link_prediction_profile.ipynb) | Canonical pairs, fingerprint, dtype, degree và hồ sơ dữ liệu cho LP | LP | B, W2.1 | `configs/data_protocol.json`, Cora qua root `data/cora/` | `results/week02/eda_summary.json`, hai hình `report/week02/figures/`, run metadata |
| [03_link_prediction/01_edge_split.ipynb](03_link_prediction/01_edge_split.ipynb) | Tạo hoặc đối chiếu split canonical, negatives 1:1 | LP | B, W2.2 | Cora, data protocol; split hiện có nếu có | `artifacts/splits/cora_lp_v1/{edges.npz,manifest.json}` nếu chưa tồn tại; metrics/run metadata |
| [03_link_prediction/02_leakage_audit.ipynb](03_link_prediction/02_leakage_audit.ipynb) | Audit từ đĩa, tái lập seed, đối chứng lỗi leakage | LP | B, W2.3 | Cora, data protocol, saved NPZ/manifest | `results/week02/leakage_audit.json`, run checks |
| [03_link_prediction/03_cosine_baseline.ipynb](03_link_prediction/03_cosine_baseline.ipynb) | Toy checks và cosine validation; kiểm tra tái lập khi có run trước | LP | B, W3.1 | Cora, saved split, audit đã pass, data/cosine configs | `results/week03/cosine_validation.json`, `runs/<id>/validation_predictions.npz`, run metadata |
| [03_link_prediction/04_cosine_analysis.ipynb](03_link_prediction/04_cosine_analysis.ipynb) | Phân bố score và phân tích kết quả validation | LP | B, W3.2 | Cosine config, selected result và prediction NPZ đúng run ID | `report/week03/figures/cosine_score_distribution.png`, phân tích inline |
| [04_framework_validation/01_runtime_imports.ipynb](04_framework_validation/01_runtime_imports.ipynb) | Tham chiếu import config, seed, hash và publication helpers | Chung | B, helper Tuần 2–4 | Editable package, dependencies | Chỉ import; không ghi file |
| [04_framework_validation/02_graph_ops_imports.ipynb](04_framework_validation/02_graph_ops_imports.ipynb) | Tham chiếu import NumPy edge/split/audit helpers | LP / framework | B, helper Tuần 2–4 | Editable package, dependencies | Chỉ import; không tải Cora hoặc tạo split |
| [04_framework_validation/03_metrics_imports.ipynb](04_framework_validation/03_metrics_imports.ipynb) | Tham chiếu import cosine và ROC-AUC/AP | LP | B, helper Tuần 3 | Editable package, dependencies | Chỉ import; không chấm điểm |
| [04_framework_validation/04_training_imports.ipynb](04_framework_validation/04_training_imports.ipynb) | Tham chiếu import callback trainer và EarlyStopping | Framework | B, helper Tuần 4 | Editable package, dependencies | Chỉ import; không huấn luyện |
| [04_framework_validation/05_graph_utilities_and_message_passing.ipynb](04_framework_validation/05_graph_utilities_and_message_passing.ipynb) | Toy edge utilities, ProjectedSum, tham chiếu vòng lặp, gradient/permutation | Framework | B, W4.1 | `configs/training_demo.json`, toy tensors; không cần Cora | `results/week04/graph_checks.json`, run metadata |
| [04_framework_validation/06_training_framework_validation.ipynb](04_framework_validation/06_training_framework_validation.ipynb) | Toy regression, early stopping và restore best checkpoint | Framework | B, W4.2 | Training demo config, synthetic train/validation tensors | `results/week04/training_checks.json`; history/checkpoint trong hai run local |
| [05_model_design/01_gat_design_and_checkin.ipynb](05_model_design/01_gat_design_and_checkin.ipynb) | Thiết kế GAT, toy grouped softmax, tổng hợp bằng chứng check-in | GAT / LP | B, W4.3 | Training config; EDA, audit, cosine, graph/training summaries và split manifest | `results/week04/checks.json`, run metadata; không có GAT hoàn chỉnh |

Notebook dùng `publish_result` còn cập nhật `results/experiment_index.csv`.
Dữ liệu, saved splits, checkpoint và output giữ nguyên vị trí; chỉ notebook được di chuyển.

## Quan hệ giữa hai EDA

Hai notebook **không trùng nhau** và được giữ riêng:

- Bản A dùng NetworkX, trình bày log–log/hubs, đối chiếu ToUndirected và phân bố
  nhãn toàn graph với train mask, kèm giải thích chuẩn hóa bậc GCN.
- Bản B dùng canonical edge pairs và fingerprint để làm đầu vào kiểm toán LP,
  ghi summary có provenance và hình báo cáo. Đây là bước xuất artifact của
  pipeline LP, khác với EDA chủ yếu hiển thị inline của A.

Hợp nhất chúng sẽ phải quyết định lại cache, cách xuất kết quả và luồng giải thích.
Đợt sắp xếp này giữ toàn bộ nội dung riêng, metadata, cell IDs và outputs.
Không có notebook bị xóa hoặc gộp.

## Môi trường và cách mở

Từ repository root, dùng môi trường Python 3.11 theo [README gốc](../README.md):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Mở repository trong VS Code và chọn kernel Python của `.venv`. Nếu dùng Jupyter,
khởi động từ repository root với môi trường đó; frontend JupyterLab là lựa chọn
riêng, không phải dependency được thêm trong đợt này. Kernel có thể bắt đầu ở
root hoặc bất kỳ thư mục con trong repository, kể cả thư mục notebook mới:
bootstrap tìm root bằng các thư mục cha. Không hỗ trợ kernel bắt đầu ở một thư
mục bên ngoài checkout chỉ nhờ mở file bằng đường dẫn tuyệt đối.

Dùng **Restart Kernel and Run All** khi chủ động tái chạy một thí nghiệm.
Không cần chạy bốn notebook import trước; tất cả notebook làm việc import
module Python trực tiếp. Không dùng notebook-to-notebook imports hoặc sửa
`sys.path`.

NC giữ root Planetoid `data/` → cache `data/Cora/`; LP giữ root
`data/cora/` → cache `data/cora/Cora/`. Không di chuyển cache để đi theo vị trí
notebook. Notebook tải dữ liệu nếu cache thiếu; các notebook publish có thể
ghi đè selected summaries và hình báo cáo. Không dùng Run All cho toàn bộ cây
chỉ để kiểm tra việc đổi thư mục.

## Thứ tự đọc và phụ thuộc thực thi

1. Đọc hai EDA để phân biệt node masks, canonical pairs và raw edge columns.
   Hai EDA không phụ thuộc vào output của nhau.
2. Đọc LP theo thứ tự split → audit → cosine → analysis.
   Split cần Cora/config, không cần EDA summary. Audit cần saved split; cosine
   cần saved split và audit đã pass. Analysis cần prediction NPZ của run được
   selected JSON trỏ tới, không chỉ summary JSON. Checkout mới có thể thiếu
   prediction NPZ trong `runs/`; không coi đây là lỗi đường dẫn do di chuyển.
3. Bốn notebook import chỉ là tài liệu tham chiếu tùy chọn. Hai notebook
   framework validation chạy độc lập với Cora và độc lập với nhau.
4. GAT/check-in dùng các summary EDA, audit, cosine và hai framework validations,
   cùng manifest để đối chiếu provenance. Nó không tự thực thi notebook khác.

Giữ các ID W2.1–W4.3 để tra lời nhắc trong notebook/báo cáo. Muốn kiểm tra
reproduction của cosine, cần hai lần chạy độc lập có saved predictions; không
suy ra đã tái lập chỉ vì summary tồn tại. Split và audit notebook có bước tạo
lại mảng trong bộ nhớ để đối chiếu seed; không chạy chúng trong kiểm tra
reorganization. Test pairs chỉ được đọc để audit giao tập, không chấm điểm.

## Source, notebook và automated checks

Repository giữ package top-level `models/`, `layers/`, `train/`, `utils/`,
`tasks/`; **không có cây `src/` thứ hai**. Bốn notebook import trỏ lần lượt đến
`utils.experiment`, `utils.link_graph`, `utils.link_metrics`,
`train.callback_trainer`. Cosine orchestration ở `tasks.link_prediction`;
ProjectedSum ở `layers.projected_sum`; receiver softmax ở `utils.attention`.
Chúng là implementation chuẩn, còn notebook cung cấp giải thích/thực nghiệm.

Automated checks ở [tests/](../tests/). Sau cài editable package:
`python -m pytest --ignore=tests/test_data_loader.py -q` kiểm tra bằng dữ liệu
synthetic và saved split. Lệnh `python -m scripts.smoke` của đợt cleanup trước
chỉ dùng validation và output tạm nhưng vẫn chạy hai epoch; không cần chạy lại
cho thao tác di chuyển notebook. Kiểm tra hiện tại dùng schema, so sánh nội dung,
imports và bootstrap paths; không tái chạy thí nghiệm.

## Bảng đổi đường dẫn

Các đường dẫn dưới đây tính từ `notebooks/`. Cột cũ là thông tin lịch sử,
không phải vị trí còn được hỗ trợ để mở file.

| Đường dẫn cũ | Đường dẫn hiện tại |
|---|---|
| `01_eda_cora.ipynb` | `01_data_exploration/01_cora_structure_and_node_labels.ipynb` |
| `member_b/week02/01_cora_eda.ipynb` | `01_data_exploration/02_cora_link_prediction_profile.ipynb` |
| `member_b/week02/02_link_prediction_split.ipynb` | `03_link_prediction/01_edge_split.ipynb` |
| `member_b/week02/03_leakage_audit.ipynb` | `03_link_prediction/02_leakage_audit.ipynb` |
| `member_b/week03/01_cosine_baseline.ipynb` | `03_link_prediction/03_cosine_baseline.ipynb` |
| `member_b/week03/02_baseline_analysis.ipynb` | `03_link_prediction/04_cosine_analysis.ipynb` |
| `shared/runtime.ipynb` | `04_framework_validation/01_runtime_imports.ipynb` |
| `shared/graph_ops.ipynb` | `04_framework_validation/02_graph_ops_imports.ipynb` |
| `shared/metrics.ipynb` | `04_framework_validation/03_metrics_imports.ipynb` |
| `shared/training.ipynb` | `04_framework_validation/04_training_imports.ipynb` |
| `member_b/week04/01_graph_utilities_and_message_passing.ipynb` | `04_framework_validation/05_graph_utilities_and_message_passing.ipynb` |
| `member_b/week04/02_training_framework_validation.ipynb` | `04_framework_validation/06_training_framework_validation.ipynb` |
| `member_b/week04/03_gat_design_and_checkin.ipynb` | `05_model_design/01_gat_design_and_checkin.ipynb` |

Link trong các báo cáo đã trỏ tới notebook mới. `creator_notebook` khi tạo
**manifest mới** dùng đường dẫn mới; saved manifests, source hashes và kết quả
lịch sử giữ nguyên để không viết lại provenance. Cây thư mục trong kế hoạch
cá nhân B mô tả giai đoạn trước tích hợp; danh mục vận hành hiện tại là bảng trên.

## Xác minh đợt di chuyển

- Cả 13 notebook đạt `nbformat.validate`; code cell hợp lệ về cú pháp.
- 9 bootstrap tìm đúng root khi kernel bắt đầu ở thư mục mới; 4 notebook chỉ
  import chạy được. Tổng cộng 18 module được import thành công trong `.venv`.
- So với working tree ngay trước khi di chuyển, 7 notebook giữ nguyên từng byte.
  6 notebook chỉ đổi 5 Markdown cell hướng dẫn và 1 code cell chứa đường dẫn
  `creator_notebook`. Toàn bộ cell IDs, metadata, execution counts và outputs
  giữ nguyên; không cần xóa output vì không thay đổi phép tính hoặc số liệu in ra.
- Không mất notebook hoặc sửa file Python/config. Chỉnh sửa chưa commit từ đợt
  trước được giữ nguyên. Link local trong các tài liệu cập nhật đều resolve.
- Không thực thi cell thí nghiệm, tải dữ liệu, tạo lại split, train hoặc chấm
  test. Không chạy lại pytest trong đợt chỉ di chuyển này; kết quả test của đợt
  cleanup trước được ghi riêng trong README gốc.

Giới hạn của checkout khi xác minh: cache NC `data/Cora/` có sẵn, cache LP mặc
định `data/cora/Cora/` chưa có; prediction NPZ của cosine run lịch sử được chọn
cũng chưa có trong `runs/`. Đây là input cần chuẩn bị cho việc chạy lại notebook,
không phải artifact bị mất trong đợt di chuyển. Metadata/output đang lưu trong
notebook là kết quả lịch sử, không phải bằng chứng thực thi lại tại vị trí mới.
