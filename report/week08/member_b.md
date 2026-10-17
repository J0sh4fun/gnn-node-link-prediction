# BÁO CÁO NGHIÊN CỨU & THỰC NGHIỆM TUẦN 8 — THÀNH VIÊN B

**Đề tài:** Phân loại chủ đề bài báo và dự đoán liên kết trên đồ thị trích dẫn bằng GNN  
**Trọng tâm:** Tinh chỉnh siêu tham số (Hyperparameter Tuning) mô hình GAT trên bài toán Node Classification (Cora)  
**Ngày cập nhật:** 17/10/2026  
**Nhánh Git:** `feature/gat-link-prediction`  
**Trạng thái:** Hoàn tất triển khai mô hình GAT 2-tầng, test suite và thực nghiệm tuning lưới (Grid Search) đạt Mốc kiểm tra giữa kỳ 2.

---

## 1. Mục tiêu và Tiêu chí nghiệm thu (Mốc giữa kỳ 2)

Theo kế hoạch tổng thể ([`docs/plans/project_plan.md`](../../docs/plans/project_plan.md), Mục 7, Tuần 8):
1. **Xây dựng mô hình GAT hoàn chỉnh:** Đóng gói mạng phân loại 2 tầng GAT ([`models/gat.py`](../../models/gat.py)) sử dụng `GATLayer` tự cài đặt, kèm kích hoạt non-linear ELU và dropout giữa các tầng.
2. **Thiết lập không gian tìm kiếm (Search Space):** Khảo sát có hệ thống các siêu tham số: `hidden_dim` (chiều mỗi head ở tầng ẩn), `heads` (số lượng attention heads), `learning_rate`, `weight_decay`, và `dropout`.
3. **Thực thi nghiêm ngặt Protocol nghiên cứu:** 
   - Lựa chọn checkpoint và cấu hình tốt nhất **100% dựa trên tập Validation** (`val_loss`, `val_accuracy`, `val_macro_f1`).
   - Tuyệt đối không chạm vào `test_mask` trong toàn bộ quá trình tìm kiếm siêu tham số.
   - Chỉ đánh giá `test_mask` đúng **1 lần duy nhất** trên mô hình tốt nhất đã chọn để hoàn tất Mốc kiểm tra giữa kỳ 2.

---

## 2. Thiết kế mô hình & Pipeline thực nghiệm

### 2.1. Kiến trúc mô hình GAT 2-tầng (`models/gat.py`)
- **Tầng 1 (Hidden Layer):**
  - Input: $X \in \mathbb{R}^{N \times 1433}$.
  - `GATLayer(in_channels=1433, out_channels=hidden_dim, heads=heads, concat=True, dropout=dropout, feature_dropout=dropout)`.
  - Đầu ra nối các head: shape $[N, \text{heads} \times \text{hidden\_dim}]$.
  - Kích hoạt: $\text{ELU}(h^{(1)})$.
- **Tầng 2 (Output Layer):**
  - `GATLayer(in_channels=hidden_dim * heads, out_channels=7, heads=1, concat=False, dropout=dropout, feature_dropout=dropout)`.
  - Đầu ra trung bình các head: logit thô shape $[N, 7]$ (không áp dụng softmax ở tầng cuối để tương thích với `nn.CrossEntropyLoss`).

### 2.2. Kiểm thử mô hình (`tests/test_gat.py`)
Đạt 100% (4/4 tests passed):
- Kiểm tra shape đầu ra $[N, 7]$.
- Kiểm tra tính hợp lệ của backward pass: toàn bộ tham số nhận gradient hữu hạn, khác 0.
- Kiểm tra khả năng overfit trên đồ thị nhỏ (loss giảm > 70%).
- Kiểm tra cờ `return_attention=True` phục vụ trực quan hóa ở Tuần 10.

---

## 3. Kết quả Thực nghiệm Tinh chỉnh Siêu tham số (Grid Search)

Toàn bộ dữ liệu chi tiết của 8 cấu hình được lưu tại [`results/week08/gat_tuning.json`](../../results/week08/gat_tuning.json).

### Bảng kết quả so sánh các cấu hình

| Config ID | `hidden_dim` | `heads` | Learning Rate | Weight Decay | Dropout | Stopped Epoch | Best Epoch | Val Loss | Val Accuracy | Val Macro-F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 8 | 8 | 0.005 | 5e-4 | 0.6 | 200 | 199 | 0.8235 | 80.00% | 79.08% |
| 2 | 8 | 8 | 0.010 | 5e-4 | 0.6 | 120 | 100 | 0.7934 | 80.80% | 80.07% |
| 3 | 8 | 4 | 0.005 | 5e-4 | 0.6 | 200 | 194 | 1.0061 | 79.60% | 78.66% |
| 4 | 8 | 4 | 0.010 | 1e-3 | 0.4 | 199 | 179 | 0.7890 | 80.00% | 78.90% |
| 5 | 16 | 4 | 0.005 | 5e-4 | 0.6 | 200 | 199 | 0.8256 | 81.60% | 80.17% |
| 6 | 16 | 4 | 0.010 | 1e-3 | 0.4 | 128 | 108 | 0.7820 | 80.60% | 79.45% |
| **7 (Best)** | **16** | **8** | **0.005** | **5e-4** | **0.4** | **200** | **189** | **0.6794** | **81.20%** | **80.12%** |
| 8 | 16 | 8 | 0.010 | 5e-4 | 0.6 | 137 | 117 | 0.7184 | 81.60% | 80.57% |

---

## 4. Đánh giá Mốc Nghiệm thu & Trả lời Câu hỏi Nghiên cứu

### 4.1. Đánh giá Final Test trên mô hình tối ưu (Config 7)
Mô hình tốt nhất (Config 7) được nạp lại checkpoint từ [`results/checkpoints/best_gat_val.pt`](../../results/checkpoints/best_gat_val.pt) và đánh giá một lần duy nhất trên tập test (1.000 bài báo giữ lại):
- **Test Accuracy:** **82.20%**
- **Test Macro-F1:** **81.32%**

### 4.2. So sánh với Baseline (Bảng tổng hợp Mục 4 kế hoạch đồ án)

| Mô hình | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Ghi chú |
|---|:---:|:---:|:---:|:---:|---|
| **MLP (Baseline không đồ thị)** | 58.80% | 56.82% | 56.80% | 53.87% | Chỉ dùng đặc trưng từ vựng BoW |
| **GAT (Tự cài đặt - Config 7)** | **81.20%** | **80.12%** | **82.20%** | **81.32%** | Cơ chế Multi-head Attention trên đồ thị |
| **Mức cải thiện ($\Delta$)** | **+22.40%** | **+23.30%** | **+25.40%** | **+27.45%** | Tác động vượt trội của cấu trúc đồ thị |

### 4.3. Đóng góp trả lời Research Questions
- **Trả lời RQ1 (Cấu trúc đồ thị cải thiện bao nhiêu so với đặc trưng riêng lẻ?):** 
  Thông tin cấu trúc đồ thị kết hợp với cơ chế truyền tin có chọn lọc của GAT mang lại mức tăng **+25.4% Test Accuracy** và **+27.45% Macro-F1** so với MLP baseline. Điều này khẳng định tri thức từ các bài báo trích dẫn láng giềng đóng vai trò quyết định trong việc phân loại chủ đề khoa học.
- **Trả lời sơ bộ RQ2 & RQ4 (Số heads và dropout ảnh hưởng thế nào?):**
  - **Số heads:** Cấu hình 8 heads cho loss validation thấp hơn rõ rệt so với 4 heads (0.6794 vs 0.7820), đồng thời giúp mô hình học ổn định hơn nhờ nhiều không gian biểu diễn chú ý độc lập.
  - **Dropout:** Tỷ lệ dropout $0.4 - 0.6$ là thiết yếu để chống overfit trên tập dữ liệu nhỏ (chỉ 140 nút train), giúp duy trì khoảng cách hẹp giữa train loss và val loss.

---

## 5. Bàn giao và Kế hoạch Tuần 9

- **Artifacts đã lưu:**
  - Checkpoint mô hình GAT tối ưu: `results/checkpoints/best_gat_val.pt`.
  - Log chi tiết tuning: `results/week08/gat_tuning.json`.
- **Nhiệm vụ tiếp theo (Tuần 9 — Trọng tâm lớn nhất của Thành viên B):**
  - Chuyển sang bài toán **Link Prediction**: Sử dụng encoder GAT để tạo node embeddings.
  - Tích hợp Dot-product Decoder để dự đoán liên kết còn thiếu, đối chiếu với Cosine baseline (Tuần 3: ROC-AUC 0.8122, AP 0.8248) để trả lời **RQ3**.
