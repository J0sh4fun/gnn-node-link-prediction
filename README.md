# GNN Node Classification & Link Prediction on Cora

Dự án nghiên cứu và triển khai Graph Neural Networks (GCN & GAT) cho hai bài toán: **Node Classification** (Thành viên A) và **Link Prediction** (Thành viên B) trên tập dữ liệu Cora.

---

## 1. Cấu trúc phân công

- **Thành viên A (Node Classification & GCN):**
  - Data loading với chuẩn hóa $L_1$ node features: `utils/data_loader.py`.
  - Mô hình baseline MLP: `models/mlp.py`, pipeline huấn luyện: `train/train_mlp.py`.
  - Khung huấn luyện module hóa `NodeClassificationTrainer`: `train/trainer.py`.
  - Thư viện đánh giá và độ đo: `utils/evaluate.py`.
  - Báo cáo lý thuyết phổ và công thức GCN: `report/GCN_Spectral_to_Formula_vi.md`.
  - Test suite hoàn chỉnh: `tests/`.

- **Thành viên B (Link Prediction & GAT):**
  - Phân tích khám phá dữ liệu Cora EDA: `notebooks/member_b/week02/01_cora_eda.ipynb`.
  - Chia tập cạnh cố định (85% train, 5% val, 10% test) và negative sampling 1:1: `artifacts/splits/cora_lp_v1/`.
  - Kiểm toán rò rỉ dữ liệu (leakage audit): `notebooks/member_b/week02/03_leakage_audit.ipynb`.
  - Baseline Cosine Similarity trên node features (Validation ROC-AUC 0.81217 / AP 0.82480): `notebooks/member_b/week03/`.
  - Thiết kế và kiểm thử prototype GAT, neighborhood attention & training framework: `notebooks/member_b/week04/`.
  - Báo cáo tiến độ theo tuần: `report/week01/` đến `report/week04/`.

---

## 2. Cài đặt môi trường

Cấu hình tham chiếu: **Python 3.11, Windows / Linux x86_64**.

### Trên Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Kiểm tra môi trường:
```powershell
.\.venv\Scripts\python.exe -c "import torch, torch_geometric; print('torch:', torch.__version__, 'PyG:', torch_geometric.__version__)"
```

---

## 3. Hướng dẫn chạy

### Chạy phần của Thành viên A (Node Classification):
- Chạy unit tests:
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/ -v
  ```
- Huấn luyện baseline MLP:
  ```powershell
  .\.venv\Scripts\python.exe train/train_mlp.py
  ```

### Chạy phần của Thành viên B (Link Prediction):
- Chạy tự động toàn bộ pipeline notebook:
  ```powershell
  .\.venv\Scripts\python.exe runs/run_all.py
  ```
- Hoặc mở từng notebook trong `notebooks/member_b/` bằng VS Code và chọn kernel `.venv`.

---

## 4. Graph Convention

Nhóm sử dụng Cora dưới dạng **đồ thị vô hướng**, áp dụng nhất quán cho cả GCN và GAT:

$$
\tilde{A} = A + I_N, \qquad \hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}
$$

- **Node classification:** Message passing trên toàn bộ đồ thị đã đối xứng; chỉ tính loss trên train mask.
- **Link prediction:** Tách biệt cạnh strictly: encoder adjacency chỉ chứa `train_pos` đã đối xứng hai chiều. Tuyệt đối không chứa cạnh `val` hay `test`. Negative edges lấy từ phần bù của toàn bộ đồ thị gốc, tỷ lệ 1:1 và không trùng nhau giữa các tập.
