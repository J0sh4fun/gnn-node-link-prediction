# Notebook phần thành viên B

Chạy bằng kernel Python 3.11 của `.venv`; mỗi notebook dùng **Restart Kernel and Run All**. `%run` tự nạp helper tương ứng từ `shared/`; không cần chạy helper riêng. Dữ liệu truyền qua file trên đĩa, không qua trạng thái RAM của notebook khác.

| Thứ tự | Notebook | Output |
|---|---|---|
| 1 | [W2.1 EDA](member_b/week02/01_cora_eda.ipynb) | EDA JSON, degree/class figures |
| 2 | [W2.2 Split](member_b/week02/02_link_prediction_split.ipynb) | `artifacts/splits/cora_lp_v1/` |
| 3 | [W2.3 Audit](member_b/week02/03_leakage_audit.ipynb) | Audit JSON |
| 4 | [W3.1 Cosine](member_b/week03/01_cosine_baseline.ipynb), chạy hai lần | Validation JSON, run predictions |
| 5 | [W3.2 Analysis](member_b/week03/02_baseline_analysis.ipynb) | Score figure |
| 6 | [W4.1 Utilities/MessagePassing](member_b/week04/01_graph_utilities_and_message_passing.ipynb) | Toy graph checks JSON |
| 7 | [W4.2 Framework](member_b/week04/02_training_framework_validation.ipynb) | Toy training checks JSON, local checkpoint |
| 8 | [W4.3 GAT design/check-in](member_b/week04/03_gat_design_and_checkin.ipynb) | Checks JSON |

`shared/runtime.ipynb` quản lý config/run/hash; `graph_ops.ipynb` chuẩn hóa cạnh/split/audit; `metrics.ipynb` chấm cosine/ROC-AUC/AP; `training.ipynb` quản lý early stopping/checkpoint. Cora và run history theo Git ignore; split đã audit và summary JSON được commit.
