# Kế hoạch bài tập lớn: Phân loại nút & dự đoán liên kết bằng GNN

**Đề tài:** Phân loại chủ đề bài báo & dự đoán liên kết trên đồ thị trích dẫn
**Kiến trúc:** Tự cài đặt tầng Message Passing (GCN / GAT) bằng PyTorch Geometric
**Bộ dữ liệu:** Cora (bắt buộc, ưu tiên hoàn thiện trước), Citeseer / Pubmed (mở rộng, nice-to-have)
**Thời gian:** 3 tháng (12 tuần) — Nhóm 2 người (gọi là **A** và **B**)

---

## 0. Problem Statement

### 0.1. Node Classification
- **Input:** Đồ thị trích dẫn $G=(V,E)$, mỗi nút $v \in V$ có vector đặc trưng bag-of-words $x_v \in \mathbb{R}^d$ (từ vựng bài báo); cạnh $E$ biểu diễn quan hệ trích dẫn giữa các bài báo.
- **Output:** Nhãn chủ đề $y_v \in \{1,...,C\}$ cho mỗi nút (bài báo).
- **Mục tiêu tối ưu:** Học hàm $f_\theta(X, A) \to \hat{Y}$ tối thiểu hóa Cross-Entropy Loss giữa $\hat{y}_v$ và nhãn thật trên tập train, đánh giá trên tập test theo Accuracy và Macro-F1.

### 0.2. Link Prediction
- **Input:** Đồ thị $G$ với một phần cạnh bị ẩn đi (held-out edges), đặc trưng nút $X$.
- **Output:** Xác suất tồn tại liên kết $\hat{p}(u,v) \in [0,1]$ cho một cặp nút $(u,v)$ bất kỳ.
- **Mục tiêu tối ưu:** Học encoder $z_v = f_\theta(X, A_{train})$ và decoder $\hat{p}(u,v) = g(z_u, z_v)$ tối thiểu hóa Binary Cross-Entropy giữa nhãn cạnh thật (positive) và cạnh giả (negative sampled), đánh giá bằng ROC-AUC và Average Precision (AP) trên cạnh test chưa từng thấy trong quá trình train.

> Hai bài toán dùng chung encoder GCN/GAT tự cài đặt nhưng là hai pipeline huấn luyện/đánh giá **độc lập**, không train chung một lúc.

---

## 1. Research Questions

Toàn bộ thực nghiệm và ablation trong kế hoạch đều nhằm trả lời các câu hỏi sau:

- **RQ1:** Thông tin cấu trúc đồ thị (graph structure) có cải thiện hiệu năng phân loại so với chỉ dùng đặc trưng từ vựng (MLP) không, và cải thiện bao nhiêu?
- **RQ2:** Cơ chế attention của GAT có mang lại lợi ích gì so với cơ chế tổng hợp chuẩn hóa theo bậc nút, không học (degree-normalized, non-learned aggregation) của GCN, trên cả hai bài toán?
- **RQ3:** Embedding học được từ GCN/GAT có hữu ích cho việc dự đoán liên kết còn thiếu tốt hơn phương pháp không học (cosine similarity trên đặc trưng gốc) không?
- **RQ4:** Mô hình nhạy cảm thế nào với số tầng (layer), dropout, và số attention head — có xảy ra hiện tượng over-smoothing khi tăng độ sâu không?

Mỗi thí nghiệm ở tuần 7–10 sẽ được gắn nhãn rõ nó phục vụ RQ nào trong báo cáo.

---

## 2. Đặc tả kỹ thuật (Technical Specification)

### 2.1. "Tự cài đặt" nghĩa là gì
Nhóm được phép kế thừa `torch_geometric.nn.MessagePassing` làm lớp cơ sở (base class) để tận dụng cơ chế `propagate()`, nhưng **bắt buộc tự viết tay** các phần sau — không dùng `GCNConv`/`GATConv` có sẵn:
- `forward()`: chuẩn bị input, gọi propagate.
- `message()`: quy tắc tạo thông điệp giữa các nút láng giềng (bao gồm hệ số chuẩn hóa của GCN hoặc attention coefficient của GAT).
- `aggregate()` (nếu override): quy tắc tổng hợp thông điệp.
- Với GAT: công thức tính attention score, softmax theo láng giềng, multi-head logic (concat/average).
- Với GCN: công thức chuẩn hóa ma trận kề (normalization).

### 2.2. Công thức GCN (Thành viên A cài đặt)

$$\tilde{A} = A + I$$
$$\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$$
$$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$
$$H^{(l+1)} = \sigma\left(\hat{A} H^{(l)} W^{(l)}\right)$$

Lưu ý: self-loop ($A+I$) phải được thêm **trước** khi tính $\tilde{D}$, đây là lỗi hay quên nhất khi tự cài.

**⚠️ Lưu ý về implementation — không tạo dense adjacency $N \times N$:** công thức trên viết ở dạng ma trận để dễ hiểu về mặt lý thuyết, nhưng khi cài đặt **không được** dựng ma trận dense $\hat{A} \in \mathbb{R}^{N \times N}$ (với Cora $N \approx 2708$ vẫn còn ổn, nhưng đây là thói quen sai sẽ không scale và không đúng tinh thần "message passing"). Cách cài đúng:
- Giữ đồ thị ở dạng `edge_index` (sparse, danh sách cạnh) như PyTorch Geometric cung cấp.
- Tính hệ số chuẩn hóa **theo từng cạnh** (edge-wise normalization): với cạnh $(i,j)$, hệ số là $\frac{1}{\sqrt{\tilde{D}_{ii}} \sqrt{\tilde{D}_{jj}}}$, tính từ bậc (degree) của từng nút qua `degree()` của PyTorch Geometric, không qua nhân ma trận dense.
- Nhân hệ số này vào `message()` khi lan truyền thông điệp qua `propagate()` (dùng cơ chế scatter/gather sparse có sẵn của `MessagePassing`), tương đương về mặt toán học với $\hat{A}H^{(l)}W^{(l)}$ nhưng chạy trên biểu diễn sparse.

### 2.3. Công thức GAT (Thành viên B cài đặt)

Với mỗi cặp nút láng giềng $(i,j)$, $j \in \mathcal{N}(i)$:

$$e_{ij} = \text{LeakyReLU}\left(a^T [W h_i \, \| \, W h_j]\right)$$
$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$
$$h_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j\right)$$

Với multi-head attention ($K$ head):
- Tầng ẩn (hidden layer): **concatenate** $K$ head: $h_i' = \|_{k=1}^{K} \sigma\left(\sum_j \alpha_{ij}^k W^k h_j\right)$
- Tầng cuối (output layer): **average** $K$ head thay vì concat, để giữ số chiều output đúng bằng số lớp.

**Self-loop trong GAT:** GAT mặc định thêm self-loop để mỗi node tham gia vào attention neighborhood của chính nó (tức $i \in \mathcal{N}(i)$ khi tính $\alpha_{ij}$); convention này được giữ cố định trong toàn bộ các thí nghiệm, trừ khi ablation nói khác.

### 2.4. Directed vs Undirected
Đồ thị trích dẫn về bản chất có hướng (bài báo A trích dẫn bài báo B), nhưng benchmark Cora/Citeseer/Pubmed chuẩn thường được xử lý như đồ thị **vô hướng** (symmetric adjacency) để phù hợp với công thức chuẩn hóa GCN gốc.
→ **Quyết định của nhóm:** dùng đồ thị vô hướng (symmetrize edge_index bằng `to_undirected()`), áp dụng nhất quán cho cả node classification lẫn link prediction, cả GCN lẫn GAT. Quyết định này được ghi vào README và không thay đổi giữa các thí nghiệm.

### 2.5. Loss Function
- Node Classification: **Cross-Entropy Loss** trên tập train (masked).
- Link Prediction: **Binary Cross-Entropy Loss** trên cặp cạnh positive + negative sampled.

---

## 3. Protocol thực nghiệm

### 3.1. Protocol cho Node Classification
- Dùng split chuẩn Planetoid (train: 20 nút/lớp, val: 500 nút, test: 1000 nút).
- **Validation set** chỉ dùng để: early stopping và chọn hyperparameter tốt nhất.
- **Test set** chỉ được chạy **một lần duy nhất**, ở bước cuối cùng sau khi đã chốt mô hình — không dùng test set để tinh chỉnh bất kỳ thứ gì trong quá trình phát triển.

### 3.2. Protocol cho Link Prediction (bao gồm Leakage Prevention)
- Chia cạnh: **85% train / 5% validation / 10% test** (tỷ lệ tham khảo, có thể điều chỉnh nhưng phải cố định trước khi chạy thí nghiệm chính).
- **Negative sampling — phải lấy trên full graph, không phải train graph:**
  - Negative edge (cặp nút không có liên kết thật) phải được kiểm tra là **không tồn tại trong toàn bộ đồ thị gốc** (train + validation + test edges), chứ không chỉ kiểm tra "không có trong train graph".
  - Lý do: nếu chỉ loại trừ trên train graph, một negative edge có thể vô tình trùng với một **held-out positive edge** (edge đã bị ẩn sang tập val/test) → model bị dạy sai là "không có liên kết" cho một cặp nút thực ra có liên kết thật, gây nhiễu nhãn.
  - Thực hiện: dùng `negative_sampling()` của PyTorch Geometric với `edge_index` là **toàn bộ đồ thị gốc** (trước khi split) để loại trừ, chứ không truyền `train_edge_index`. Sample riêng negative set cho từng tập train/val/test, đảm bảo tỷ lệ 1:1 với positive edge tương ứng của tập đó và không trùng lặp giữa các tập.
- **⚠️ Mục Leakage Prevention (quan trọng nhất về mặt kỹ thuật):**
  - Đồ thị dùng làm input cho encoder GCN/GAT khi **train** chỉ được chứa **train edges** — loại bỏ hoàn toàn validation edges và test edges khỏi `edge_index` dùng cho message passing.
  - Khi **đánh giá trên validation** trong lúc đang chọn siêu tham số/model, encoder vẫn chỉ dùng train edges để lan truyền thông điệp (không được "nhìn thấy" edge đang được đánh giá).
  - Xem Mục 3.3 để biết chính xác encoder dùng adjacency nào khi đánh giá trên **test**.
  - Không bao giờ để test/validation edge xuất hiện trong adjacency matrix dùng để tính embedding tại thời điểm nó đang được dự đoán.

### 3.3. Final Test Protocol — chốt cách xử lý train+val trước khi test
Để tránh mâu thuẫn giữa "model được train trên train graph" và "test được phép dùng train+val edges để lan truyền", nhóm chọn **quy trình 2 bước chuẩn** (giống protocol phổ biến trong GAE/VGAE và các paper link prediction):
1. **Bước phát triển (tuning):** train model trên **train edges**, đánh giá & chọn siêu tham số tốt nhất bằng **validation set** — encoder chỉ dùng train edges để lan truyền khi đánh giá validation (đúng Mục 3.2).
2. **Bước đánh giá cuối cùng (final test, chỉ chạy 1 lần):** với cấu hình siêu tham số đã chốt, **train lại (retrain) model từ đầu trên train + validation edges** (không đụng đến test edges), sau đó dùng encoder đã retrain này (lan truyền trên train+val edges) để dự đoán trên test set.
- Việc retrain ở bước 2 phải được ghi rõ trong báo cáo và code (`train/retrain_final.py`), tách biệt khỏi vòng lặp tuning ở bước 1, để không ai nhầm rằng test dùng model chỉ được train trên train edges.

### 3.4. Kiến trúc Link Prediction
```
Graph (train edges) → GCN/GAT Encoder → Node Embedding z_v → Edge Decoder → Probability p(u,v)
```
- **Decoder mặc định:** dot-product: $\hat{p}(u,v) = \sigma(z_u^T z_v)$.
- **Decoder mở rộng (nếu còn thời gian):** MLP nhận $[z_u \| z_v]$ làm input.
- **Phạm vi dự đoán (quan trọng để tránh hiểu nhầm):** vì đồ thị được symmetrize thành vô hướng (Mục 2.4), bài toán Link Prediction ở đây dự đoán **có tồn tại quan hệ trích dẫn giữa hai bài báo hay không** ($u$ và $v$ có liên kết, bất kể chiều), **không dự đoán hướng trích dẫn** (ai trích dẫn ai, $A \to B$ hay $B \to A$). Nếu nhóm muốn mở rộng sang dự đoán có hướng, đây sẽ là một bài toán khác (directed link prediction) và không nằm trong phạm vi bắt buộc của đồ án — cần nêu rõ trong phần "Hướng phát triển" của báo cáo, không lặng lẽ trộn lẫn hai định nghĩa.

---

## 4. Metric & Bảng kết quả dự kiến

- **Node Classification:** Accuracy + **Macro-F1** (metric chính, vì các lớp trong Cora không hoàn toàn cân bằng).
- **Link Prediction:** **ROC-AUC** và **Average Precision (AP)** làm metric chính. Không dùng accuracy làm metric trung tâm cho bài toán này.

Bảng kết quả cuối cùng cần điền (thiết kế sẵn từ đầu để biết mỗi thí nghiệm đổ vào đâu):

**Node Classification**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| MLP (baseline) | | |
| GCN (tự cài) | | |
| GAT (tự cài) | | |

**Link Prediction**

| Model | ROC-AUC | AP |
|---|---|---|
| Cosine similarity (baseline) | | |
| GCN Encoder | | |
| GAT Encoder | | |

> Benchmark tham khảo (ví dụ GCN ~81–82% accuracy trên Cora) chỉ mang tính **đối chiếu**, **không phải tiêu chí pass/fail**. Tiêu chí thành công thực sự nằm ở Mục 6.

---

## 5. Acceptance Criteria — khi nào một layer/mô hình được coi là "xong"

Một tầng (`GCNLayer` hoặc `GATLayer`) được coi là hoàn thành khi đạt đủ:
- [ ] Forward pass chạy không lỗi trên graph mẫu.
- [ ] Backward pass sinh gradient hợp lệ cho toàn bộ tham số (không None).
- [ ] Không xuất hiện NaN/Inf trong output hoặc gradient.
- [ ] Output shape đúng như kỳ vọng (`[num_nodes, out_dim]`).
- [ ] Unit test so sánh với `GCNConv`/`GATConv` gốc **pass** — lưu ý: chỉ so sánh trên graph nhỏ, **cùng weight initialization và cấu hình tương đương**; nếu implementation detail khác nhau thì không kỳ vọng output giống tuyệt đối, quan trọng là logic đúng, shape đúng, gradient hợp lý và numerically consistent (không lệch quá xa).
- [ ] Loss giảm ổn định qua các epoch khi train thử trên tập nhỏ (sanity check overfit trên vài chục nút).

---

## 6. Nguyên tắc chia việc

Vì đề bài yêu cầu **tự cài đặt** cả GCN lẫn GAT, cách chia công bằng nhất là **chia theo trục kiến trúc song song với chia theo trục bài toán**, để mỗi người đều:
- Tự tay cài đặt ít nhất 1 tầng message passing từ đầu.
- Phụ trách chính 1 trong 2 bài toán (node classification / link prediction).
- Cùng tham gia đầy đủ phần báo cáo, slide và thực nghiệm so sánh.

| | **Thành viên A** | **Thành viên B** |
|---|---|---|
| Kiến trúc tự cài đặt | GCN Layer (self-loop + symmetric normalization) | GAT Layer (attention coefficients + multi-head) |
| Bài toán chính | Node Classification | Link Prediction |
| Vai trò phụ | Review code GAT của B, hỗ trợ tuning | Review code GCN của A, hỗ trợ tuning |
| Phần báo cáo | Lý thuyết GCN, thực nghiệm node classification, error analysis | Lý thuyết GAT, thực nghiệm link prediction, attention visualization |

Các phần dùng chung (data pipeline, evaluation framework, slide, viết báo cáo tổng hợp) được **chia đôi thời lượng ở mỗi tuần**, không dồn vào cuối.

---

## 7. Kế hoạch chi tiết theo tuần

### 🟦 Giai đoạn 1 — Nền tảng (Tuần 1–4)

**Tuần 1 — Khởi động & đọc tài liệu**
- Chung: Đọc paper gốc GCN (Kipf & Welling, 2017) và GAT (Veličković et al., 2018); chốt Problem Statement, Research Questions (Mục 0, 1); tạo repo Git.
- A: Tóm tắt lý thuyết spectral graph convolution → GCN.
- B: Tóm tắt lý thuyết attention mechanism trên đồ thị (GAT).
- **Deliverable:** Đề cương với Problem Statement + Research Questions + repo GitHub cấu trúc chuẩn (xem Mục 10).

**Tuần 2 — Chuẩn bị dữ liệu**
- Chung: Cài PyTorch Geometric, tải Cora (Planetoid), EDA (số nút/cạnh/nhãn/đặc trưng); quyết định directed/undirected (Mục 2.4) và ghi vào README.
- A: Pipeline load & tiền xử lý cho node classification theo protocol Mục 3.1.
- B: Pipeline edge splitting cho link prediction theo protocol Mục 3.2 (bao gồm loại test/val edge khỏi adjacency train).
- **Deliverable:** Notebook EDA + 2 data pipeline có unit test, README ghi rõ convention đồ thị.

**Tuần 3 — Baseline đơn giản**
- Chung: Xây baseline không dùng cấu trúc đồ thị để phục vụ RQ1/RQ3.
- A: Baseline MLP cho node classification (Cross-Entropy Loss, Accuracy + Macro-F1).
- B: Baseline cosine similarity cho link prediction (ROC-AUC + AP).
- **Deliverable:** Kết quả baseline điền vào bảng Mục 4, script `evaluate.py` dùng chung.

**Tuần 4 — Framework huấn luyện chung**
- Chung: Khung huấn luyện tổng quát (training loop, early stopping theo validation metric, logging, seed cố định).
- A: Viết `MessagePassingLayer` cơ sở kế thừa `MessagePassing`, chuẩn bị utility tính $\hat{A}$ (normalized adjacency) riêng cho GCN.
- B: Viết utility thêm self-loop / xử lý edge_index dùng chung cho cả 2 kiến trúc, và utility neighborhood indexing riêng cho GAT (GAT không dùng normalized adjacency theo cách của GCN).
- **Deliverable:** Framework huấn luyện tái sử dụng được, checklist review chéo code, experiment tracking schema (Mục 11) đã thiết lập.

> 🔸 **Mốc kiểm tra giữa kỳ 1 (cuối tuần 4):** Demo pipeline dữ liệu + baseline, xác nhận đi đúng hướng.

---

### 🟩 Giai đoạn 2 — Xây dựng mô hình (Tuần 5–8)

**Tuần 5 — Tự cài đặt GCN Layer**
- A (chính): Cài `GCNLayer` theo công thức Mục 2.2, không dùng `GCNConv`.
- B: Viết unit test theo Acceptance Criteria (Mục 5).
- **Deliverable:** `models/gcn_layer.py` + test pass + checklist Mục 5 đạt đủ.

**Tuần 6 — Tự cài đặt GAT Layer**
- B (chính): Cài `GATLayer` theo công thức Mục 2.3, không dùng `GATConv`.
- A: Unit test theo Acceptance Criteria, benchmark tốc độ so với `GATConv` gốc.
- **Deliverable:** `models/gat_layer.py` + test pass + checklist Mục 5 đạt đủ.

**Tuần 7 — Huấn luyện mô hình node classification**
- A (chính): Xây **riêng biệt** một mạng 2-tầng dùng `GCNLayer` và một mạng 2-tầng dùng `GATLayer` (không ghép hai loại layer vào cùng một mạng), huấn luyện cả hai trên Cora, log kết quả vào bảng Mục 4.
- B: Chạy thử trên Citeseer nếu còn thời gian (không bắt buộc); hỗ trợ debug.
- **Deliverable:** Kết quả GCN vs GAT vs MLP baseline, trả lời sơ bộ RQ1/RQ2.

**Tuần 8 — Tinh chỉnh siêu tham số**
- Chung: Grid/random search learning rate, số tầng, hidden dim, dropout, số head (GAT), weight decay — chỉ dùng validation set để chọn.
- A: Tuning GCN. B: Tuning GAT.
- **Deliverable:** Bảng siêu tham số tốt nhất, kết quả node classification cuối cùng (đối chiếu benchmark tham khảo, không coi là điều kiện pass/fail).

> 🔸 **Mốc kiểm tra giữa kỳ 2 (cuối tuần 8):** Cả 2 mô hình đạt Acceptance Criteria, có kết quả node classification hoàn chỉnh và đúng protocol (test set chỉ chạy 1 lần).

---

### 🟧 Giai đoạn 3 — Mở rộng & Hoàn thiện (Tuần 9–12)

**Tuần 9 — Bài toán Link Prediction**
- B (chính): Dùng GCN/GAT encoder đã cài để sinh embedding, dot-product decoder (mặc định) theo Mục 3.4, huấn luyện theo protocol Mục 3.2, đặc biệt kiểm tra kỹ leakage prevention.
- A: Hỗ trợ chuẩn hóa embedding, hoàn thiện `evaluate.py` cho ROC-AUC/AP.
- **Deliverable:** Kết quả link prediction GCN-encoder và GAT-encoder điền vào bảng Mục 4, trả lời RQ3.

**Tuần 10 — Ablation Study có cấu trúc & Phân tích lỗi**
- Chung: Chốt trước **3–5 ablation chính** (không mở rộng thêm để tránh quá tải trong tuần ngắn):
  1. Có/không self-loop (GCN).
  2. Số tầng: 2-layer vs 3-layer vs 4-layer — quan sát **over-smoothing** (liên hệ trực tiếp RQ4: độ chính xác có giảm khi mạng quá sâu do embedding các nút bị "trộn" đồng nhất không).
  3. Hidden dimension.
  4. Số attention head của GAT.
  5. Dot-product decoder vs MLP decoder (link prediction).
- A: Error analysis cho node classification — confusion matrix **và** phân tích theo bậc nút (node degree): nút có ít láng giềng có dễ bị phân loại sai hơn không.
- B: Trực quan hóa attention của GAT — chọn một số **node mẫu cụ thể** (không cố visualize toàn graph) và vẽ attention weight tới từng láng giềng của các node đó.
- **Deliverable:** Bộ kết quả ablation (5 mục trên) + phân tích lỗi + visualization, phục vụ trực tiếp phần Discussion và trả lời RQ4.

**Tuần 11 — Viết báo cáo & chuẩn bị slide**
- A: Giới thiệu, Problem Statement, Lý thuyết GCN, Thực nghiệm node classification, Error analysis, Kết luận.
- B: Lý thuyết GAT, Thực nghiệm link prediction, Leakage prevention, Ablation study, Attention visualization, Hướng phát triển.
- Chung: Ghép báo cáo, trả lời rõ từng Research Question ở phần Discussion, làm slide (chia đều 50/50), chuẩn bị demo.
- **Deliverable:** Bản nháp báo cáo đầy đủ + slide draft.

**Tuần 12 — Rà soát & Nộp bài**
- Chung: Đọc chéo toàn bộ báo cáo/code, dọn repo, viết README reproducibility (Mục 12), tập dượt thuyết trình.
- A & B: Luyện trả lời câu hỏi phản biện, mỗi người tự tin trình bày cả 2 phần để tránh rủi ro khi hỏi đáp.
- **Deliverable:** Báo cáo hoàn chỉnh, slide cuối, code có README, nộp đúng hạn — đối chiếu Definition of Done (Mục 13).

---

## 8. Bảng phân chia khối lượng công việc (ước tính)

| Hạng mục | A | B |
|---|---|---|
| Cài đặt kiến trúc chính (GCN/GAT) | 100% GCN | 100% GAT |
| Bài toán chính | 100% Node Classification | 100% Link Prediction |
| Review code chéo | 1 lần/tuần | 1 lần/tuần |
| Data pipeline & baseline | 50% | 50% |
| Tuning siêu tham số | 50% (GCN) | 50% (GAT) |
| Ablation study | 50% | 50% |
| Viết báo cáo | 50% | 50% |
| Slide & thuyết trình | 50% | 50% |

→ Mỗi người là **chủ lực của một kiến trúc + một bài toán**, đồng thời cùng tham gia mọi phần chung.

---

## 9. Công cụ & quy trình làm việc

- **Quản lý code:** GitHub, branch riêng (`feature/gcn`, `feature/gat`), pull request + review trước khi merge vào `main`.
- **Theo dõi tiến độ:** Kanban board (Trello/GitHub Projects): To do / Doing / Review / Done, cập nhật cuối mỗi tuần.
- **Họp nhóm:** đầu tuần (giao việc, ~20–30 phút) + cuối tuần (review kết quả).
- **Ghi log thí nghiệm:** Weights & Biases hoặc spreadsheet chung (xem schema Mục 11).

---

## 10. Cấu trúc Repository

```
project/
├── data/          # dữ liệu raw & processed
├── models/        # GCNLayer, GATLayer, các mạng hoàn chỉnh
├── train/         # script/loop huấn luyện cho từng bài toán
├── configs/        # config siêu tham số (yaml/json) cho từng run
├── utils/         # adjacency normalization, edge splitting, negative sampling...
├── tests/         # unit test cho từng layer (Acceptance Criteria)
├── results/       # log kết quả, checkpoint, hình ảnh ablation
├── notebooks/     # EDA, visualization
├── report/        # báo cáo, slide
├── requirements.txt / environment.yml
└── README.md
```

---

## 11. Experiment Tracking Schema

Mỗi lần chạy (run) bắt buộc lưu lại tối thiểu các trường sau (trong W&B hoặc 1 dòng spreadsheet):

`model | dataset | seed | learning_rate | hidden_dim | num_layers | dropout | num_heads (nếu GAT) | decoder (nếu link prediction) | validation_score | test_score (final run only)`

Lưu ý: trường `test_score` chỉ được điền cho **run cuối cùng** (final run, sau khi đã chốt mô hình và siêu tham số theo Mục 3.1/3.3) — nhất quán với nguyên tắc "test chỉ chạy một lần". Với mọi run trong quá trình tuning, để trống `test_score`.

**Quy tắc chọn checkpoint:** mô hình cuối cùng được chọn theo **validation metric tốt nhất** (best validation loss/accuracy/AUC tùy bài toán) trong quá trình huấn luyện, **không mặc định lấy epoch cuối cùng**.

---

## 12. Reproducibility Checklist

- [ ] Seed cố định (numpy, torch, torch_geometric).
- [ ] Lưu lại split (train/val/test mask hoặc edge list) dùng cho kết quả cuối cùng.
- [ ] Lưu config (siêu tham số) của run tốt nhất.
- [ ] Lưu checkpoint model tốt nhất.
- [ ] Ghi rõ phiên bản package (`requirements.txt` / `environment.yml`).
- [ ] README có lệnh cụ thể để chạy lại từng thí nghiệm từ đầu.

---

## 13. Definition of Done (toàn bộ dự án)

Dự án được coi là **hoàn thành** khi tất cả các điều kiện sau được đáp ứng:

- [ ] `GCNLayer` và `GATLayer` tự cài đặt, đạt đủ Acceptance Criteria (Mục 5).
- [ ] Node Classification hoàn chỉnh: có baseline MLP, tuân thủ protocol Mục 3.1, kết quả điền đủ bảng Mục 4.
- [ ] Link Prediction hoàn chỉnh: có baseline cosine similarity, **không có data leakage** (đã kiểm tra kỹ theo Mục 3.2), kết quả điền đủ bảng Mục 4.
- [ ] Ablation study đủ 3–5 mục đã chốt (Mục 7, tuần 10), có liên hệ rõ tới Research Questions.
- [ ] Error analysis (node classification) và attention visualization (GAT) hoàn chỉnh.
- [ ] Báo cáo trả lời rõ ràng từng Research Question ở Mục 1.
- [ ] Repo tuân thủ cấu trúc Mục 10, README đủ điều kiện reproducibility (Mục 12).
- [ ] Cora chạy hoàn chỉnh cho mọi deliverable bắt buộc; Citeseer/Pubmed (nếu có) chỉ là phần mở rộng, không ảnh hưởng đến việc đạt Definition of Done.

---

## 14. Rủi ro & phương án dự phòng

| Rủi ro | Phương án xử lý |
|---|---|
| Cài đặt tay GCN/GAT bị lỗi số học (NaN, gradient explode) | Buffer 1 tuần ở giai đoạn 2, luôn kiểm tra Acceptance Criteria trước khi coi layer là "xong" |
| Data leakage trong Link Prediction (lỗi kỹ thuật dễ mắc nhất) | Tuân thủ nghiêm ngặt Mục 3.2, review chéo bắt buộc trước khi chạy thí nghiệm chính thức |
| Một thành viên bị chậm tiến độ | Review chéo hàng tuần giúp người kia luôn nắm được code, có thể tiếp quản nếu cần |
| Kết quả không đạt benchmark tham khảo | Benchmark chỉ là đối chiếu (Mục 4), không phải tiêu chí pass/fail — tiêu chí chính là Definition of Done (Mục 13) |
| Thiếu thời gian cho Citeseer/Pubmed | Không bắt buộc — ưu tiên Cora hoàn chỉnh trước theo đúng Mục 13 |
| Ablation study quá rộng, hết thời gian ở tuần 10 | Đã chốt cứng 3–5 mục từ trước (Mục 7), không mở rộng thêm ngoài kế hoạch |

---

## 15. Mốc nộp/kiểm tra tổng hợp

| Mốc | Thời điểm | Nội dung |
|---|---|---|
| Check-in 1 | Cuối tuần 4 | Data pipeline + baseline hoàn chỉnh, convention đồ thị đã chốt |
| Check-in 2 | Cuối tuần 8 | GCN & GAT đạt Acceptance Criteria, node classification hoàn chỉnh đúng protocol |
| Check-in 3 | Cuối tuần 10 | Link prediction (không leakage) + ablation study có cấu trúc hoàn chỉnh |
| **Nộp bài cuối** | Cuối tuần 12 | Đối chiếu đầy đủ Definition of Done (Mục 13) |
