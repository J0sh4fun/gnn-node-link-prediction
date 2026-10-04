# BÁO CÁO NGHIÊN CỨU TUẦN 1 — THÀNH VIÊN B

**Đề tài:** Phân loại chủ đề bài báo và dự đoán liên kết trên đồ thị trích dẫn bằng GNN  
**Trọng tâm:** Nền tảng Graph Attention Networks (GAT) và định hướng tự cài đặt  
**Ngày biên soạn:** 27/09/2026  
**Căn cứ phạm vi:** Kế hoạch nhóm trong `Plan GNN.md`, đặc biệt các mục 0–3, 6–7 và 10–12.  
**Loại báo cáo:** Tổng hợp nghiên cứu tài liệu, phân tích kỹ thuật và ví dụ minh họa.

> Báo cáo phục vụ học tập và chuẩn bị triển khai của thành viên B. Chưa huấn luyện mô hình trên Cora, chưa đo chỉ số dự đoán và chưa tạo repository trong phạm vi công việc này. Các ví dụ nhỏ là dữ liệu tự xây dựng, không phải kết quả thực nghiệm của nhóm. Tài liệu tham khảo được dẫn bằng mã [S1]–[S12] và liệt kê ở cuối báo cáo.

## Tóm tắt

Nhiệm vụ nghiên cứu tuần 1 của B là làm rõ cách attention tổng hợp thông tin trên đồ thị, đối chiếu với GCN và chuyển công thức thành một thiết kế có thể tự cài đặt. Báo cáo thống nhất ký hiệu nút nhận–nút gửi, trình bày công thức GAT, minh họa bằng phép tính trên đồ thị ba nút và xác định các điểm kiểm tra trước khi viết model. Kết quả phục vụ trực tiếp RQ2 của nhóm về sự khác biệt giữa GAT và GCN, đồng thời đặt nền tảng cho RQ3 về chất lượng embedding trong dự đoán liên kết.

## 1. Mục tiêu và phạm vi tuần 1

Theo kế hoạch gốc, A nghiên cứu GCN; B nghiên cứu attention trên đồ thị và về sau phụ trách GAT cùng Link Prediction. Hai thành viên cùng thống nhất bài toán, câu hỏi nghiên cứu và cấu trúc repository.

Đầu ra tri thức của B trong tuần này gồm:

1. Giải thích được node, edge, node feature, embedding và message passing.
2. Đọc được công thức GCN để trao đổi và review với A.
3. Phân biệt attention score, attention coefficient và xác suất tồn tại cạnh.
4. Tính được một bước GAT single-head trên đồ thị nhỏ.
5. Theo dõi được kích thước tensor khi chuyển sang multi-head.
6. Biết ánh xạ công thức sang `MessagePassing` mà không dùng `GATConv` làm model chính.
7. Hiểu ranh giới dữ liệu được phép sử dụng trong hai bài toán của nhóm.

Tuần 1 tập trung nghiên cứu. Pipeline chia cạnh thuộc tuần 2, cosine baseline thuộc tuần 3, utility và framework thuộc tuần 4, còn tầng GAT hoàn chỉnh thuộc tuần 6. Các thiết kế dưới đây là chuẩn bị cho những mốc đó.

## 2. Phát biểu bài toán và câu hỏi nghiên cứu

### 2.1. Biểu diễn bài toán

Đặt $G=(V,E)$, $N=|V|$, $X\in\mathbb{R}^{N\times F}$ là ma trận đặc trưng. Mỗi hàng $x_i$ biểu diễn nội dung một bài báo. Trong phạm vi nhóm, một liên kết biểu diễn quan hệ trích dẫn **bất kể chiều**, vì đồ thị được quy ước vô hướng.

| Thành phần | Node Classification | Link Prediction |
|---|---|---|
| Đơn vị dự đoán | Một nút | Một cặp nút |
| Mục tiêu | Chủ đề bài báo | Có liên kết giữa hai bài báo hay không |
| Đầu ra | Vector điểm cho $C$ lớp | Điểm liên kết hoặc xác suất nhị phân |
| Loss theo kế hoạch | Cross-Entropy trên nút train | Binary Cross-Entropy trên cặp positive/negative |
| Chỉ số theo kế hoạch | Accuracy, Macro-F1 | ROC-AUC, Average Precision |
| Phần B chịu trách nhiệm chính | Encoder GAT, hỗ trợ A | Pipeline dự đoán liên kết |

Hai bài toán sử dụng lại kiến trúc encoder, nhưng huấn luyện và đánh giá bằng **hai pipeline độc lập**. “Dùng chung encoder” không có nghĩa bắt buộc dùng chung trọng số đã học hoặc huấn luyện đa nhiệm.

### 2.2. Câu hỏi nghiên cứu của nhóm

| Mã | Câu hỏi | Liên hệ với nghiên cứu tuần 1 của B |
|---|---|---|
| RQ1 | Cấu trúc đồ thị giúp phân loại tốt hơn đặc trưng riêng lẻ bao nhiêu? | Hiểu thông tin bổ sung đến từ láng giềng |
| RQ2 | Attention của GAT đem lại lợi ích gì so với chuẩn hóa theo bậc của GCN? | Trọng tâm lý thuyết và đối chiếu kiến trúc |
| RQ3 | Embedding GNN dự đoán liên kết tốt hơn cosine trên đặc trưng gốc không? | Phân biệt encoder, embedding và decoder |
| RQ4 | Độ sâu, dropout và số head ảnh hưởng thế nào? | Chuẩn bị biến kiểm soát cho ablation |

**Giả thuyết làm việc:** trọng số attention có thể giúp phân biệt mức đóng góp của các láng giềng. Đây là động cơ để thử nghiệm, chưa phải kết luận rằng GAT chắc chắn tốt hơn GCN.

## 3. Nền tảng dữ liệu đồ thị và Cora

### 3.1. Các khái niệm cần phân biệt

| Thuật ngữ | Ý nghĩa trong đồ án | Ví dụ diễn giải |
|---|---|---|
| Node | Một bài báo | Bài báo mang mã số 42 |
| Edge | Quan hệ trích dẫn đã quy về vô hướng | Hai bài báo có liên hệ trích dẫn |
| Node feature | Vector đầu vào mô tả bài báo | Đặc trưng từ vựng |
| Label | Chủ đề cần dự đoán | Một trong các lớp của Cora |
| Embedding | Vector do encoder tính ra | Biểu diễn kết hợp nội dung và láng giềng |
| Neighborhood | Tập nút gửi thông tin đến một nút | Các bài liên kết với bài đang xét |
| Self-loop | Cạnh từ nút về chính nó | Bài báo giữ một đường truyền thông tin của bản thân |

Node ID chỉ dùng định danh. Không mặc định dùng ID như đặc trưng số có ý nghĩa. Ví dụ, bài 42 và bài 43 không nhất thiết có nội dung gần nhau hơn bài 42 và bài 900.

### 3.2. Dữ liệu tham chiếu và cách đếm cạnh

Tài liệu `Planetoid` của PyG công bố Cora có 2.708 nút, 1.433 đặc trưng, 7 lớp và 10.556 mục cạnh trong biểu diễn của thư viện. `split="public"` sử dụng phân hoạch công khai cố định. [S5]

| Thuộc tính | Giá trị tham chiếu cho thiết kế |
|---|---:|
| $N$ | 2.708 |
| $F$ | 1.433 |
| $C$ | 7 |
| `x.shape` | `[2708, 1433]` |
| `y.shape` | `[2708]` |
| Nút train theo kế hoạch | 140 = 20 × 7 |
| Nút validation theo kế hoạch | 500 |
| Nút test theo kế hoạch | 1.000 |

Paper GAT ghi 5.429 cạnh trong bảng dữ liệu, còn tài liệu PyG ghi 10.556. [S1, S5] Không thể giải thích chênh lệch này chỉ bằng phép nhân đôi, bởi $2\times5429\ne10556$. Tuần 2 phải ghi rõ phiên bản, tiền xử lý và đếm thực tế; không chép một con số rồi coi đó là kết quả EDA.

Quy ước đếm đề xuất của báo cáo: với đồ thị đã đối xứng, loại trùng và không có self-loop, mỗi cạnh vô hướng tương ứng hai cột trong `edge_index`. Khi đủ điều kiện này, 10.556 cột tương ứng 5.278 cặp vô hướng. Self-loop thêm để tính GAT phải được báo riêng.

### 3.3. Đặc trưng và nhãn không cùng vai trò

`Data` của PyG thường lưu `x`, `edge_index`, `y` cùng các mask. `edge_index` là danh sách chỉ số cạnh dạng COO, không phải ma trận kề đặc có kích thước $N\times N$. [S6]

Trong thiết kế nhóm, `x` là thông tin đầu vào; `y` phục vụ tính loss và đánh giá. Không ghép nhãn chủ đề thật vào `x`. Một nút chưa có nhãn train vẫn có thể cung cấp đặc trưng cho láng giềng trong thiết lập transductive của bài toán phân loại.

## 4. Message passing và nền tảng GCN

### 4.1. Từ đồ thị đến phép tổng hợp

Khung message passing có thể viết:

$$
m_{j\to i}^{(l)}=\phi^{(l)}(h_i^{(l)},h_j^{(l)}),
\qquad
h_i^{(l+1)}=\gamma^{(l)}\left(h_i^{(l)},\bigoplus_{j\in\mathcal N(i)}m_{j\to i}^{(l)}\right).
$$

$\phi$ tạo thông điệp; $\bigoplus$ tổng hợp, chẳng hạn bằng tổng; $\gamma$ tạo biểu diễn mới. PyG cung cấp các điểm mở rộng `message`, `aggregate`, `update`, còn `propagate` điều phối việc truyền tin. [S3]

Một ví dụ tự xây dựng: nếu nút 0 nhận hai vector $(1,2)$ và $(3,0)$ thì tổng là $(4,2)$, trung bình là $(2,1)$. Đổi thứ tự hai vector không thay đổi hai kết quả này. Ngược lại, nối trực tiếp hai vector tạo độ dài phụ thuộc số láng giềng, không phù hợp làm đầu ra cố định cho các nút có bậc khác nhau.

### 4.2. GCN để đối chiếu với GAT

GCN của Kipf và Welling sử dụng:

$$
\tilde A=A+I,\qquad \tilde D_{ii}=\sum_j\tilde A_{ij},
$$

$$
H^{(l+1)}=\sigma\left(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)}\right).
$$

Theo từng cạnh, hệ số tổng hợp là $1/\sqrt{\tilde d_i\tilde d_j}$. Hệ số này phụ thuộc bậc nút; ma trận biến đổi $W$ vẫn được học. Vì vậy, phát biểu “GCN không học trọng số” là sai: phần không học ở đây là hệ số chuẩn hóa theo bậc. [S2]

**Suy luận áp dụng:** nếu hai nút gửi có cùng bậc và cùng gửi đến một nút nhận, GCN chuẩn gán cùng hệ số chuẩn hóa cho chúng, dù đặc trưng khác nhau. Điều này tạo cơ sở để hỏi liệu một hệ số phụ thuộc đặc trưng có hữu ích hơn hay không. Nhóm phải dùng thực nghiệm để trả lời.

| Tiêu chí | GCN chuẩn | GAT gốc |
|---|---|---|
| Phép biến đổi đặc trưng | Có tham số học | Có tham số học |
| Hệ số của láng giềng | Chuẩn hóa theo bậc | Tính từ đặc trưng và tham số attention |
| Nhiều head | Không phải thành phần chuẩn | Có |
| Phần cần kiểm tra khi tự cài | Degree sau self-loop, hệ số cạnh | Nhóm softmax, chiều cạnh, head |

Bảng là đối chiếu phục vụ thiết kế từ [S1, S2], không phải bảng đánh giá hiệu năng.

## 5. Công thức GAT và cách đọc

### 5.1. Quy ước ký hiệu

Trong toàn báo cáo, **$i$ là nút nhận, $j$ là nút gửi**; $\mathcal N^+(i)=\mathcal N(i)\cup\{i\}$. Một head có $F'$ chiều đầu ra. Vector nút trong công thức là vector cột; tensor thực thi chứa mỗi nút trên một hàng.

| Ký hiệu | Vai trò | Kích thước single-head |
|---|---|---|
| $h_i$ | Đặc trưng nút | $F$ |
| $W$ | Biến đổi tuyến tính | $F'\times F$ |
| $g_i=Wh_i$ | Đặc trưng đã chiếu | $F'$ |
| $a$ | Tham số attention | $2F'$ |
| $e_{ij}$ | Score trước softmax | Một số thực |
| $\alpha_{ij}$ | Hệ số sau softmax | Một số không âm |
| $h'_i$ | Biểu diễn đầu ra | $F'$ |

### 5.2. Bốn phép tính cốt lõi

Công thức GAT single-head trong paper: [S1, mục 2.1]

$$
g_i=Wh_i,
$$

$$
e_{ij}=\operatorname{LeakyReLU}\left(a^T[g_i\Vert g_j]\right),
$$

$$
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{t\in\mathcal N^+(i)}\exp(e_{it})},
$$

$$
h'_i=\sigma\left(\sum_{j\in\mathcal N^+(i)}\alpha_{ij}g_j\right).
$$

Đọc theo thứ tự: biến đổi đặc trưng; chấm điểm các nguồn tin hợp lệ; chuẩn hóa mức đóng góp; cộng các vector đã nhân hệ số. $W$ và $a$ được tối ưu cùng mô hình. LeakyReLU dùng nhánh âm có hệ số 0,2 trong paper. [S1]

**Phân biệt ba đại lượng:** $e_{ij}$ chưa chuẩn hóa; $\alpha_{ij}$ là mức đóng góp trong một neighborhood; xác suất tồn tại cạnh là đầu ra của decoder Link Prediction. Không thay decoder bằng $\alpha_{ij}$.

### 5.3. Graph structure đi vào attention ở đâu?

Masked attention giới hạn các cặp được tính theo neighborhood. Tham số chia sẻ giữa các cạnh, không cấp một bộ tham số riêng cho từng node ID. Trang giải thích của tác giả mô tả đây là cơ chế tổng hợp cục bộ, có thể dùng cho đồ thị chưa gặp khi huấn luyện. [S7]

**Suy luận cho đồ án:** với một cặp không thuộc neighborhood hiện tại, GAT chuẩn không tự sinh thêm cạnh chỉ vì hai bài báo giống nhau. Phần dự đoán cạnh thiếu do decoder thực hiện sau khi encoder đã tạo embedding.

### 5.4. Softmax theo neighborhood và độ ổn định số

Mỗi nút nhận, mỗi head có một nhóm softmax riêng. Triển khai sparse softmax của PyG gom giá trị theo `index`, trừ cực đại của từng nhóm rồi mới lấy hàm mũ. [S8]

$$
m_i=\max_{t\in\mathcal N^+(i)}e_{it},\qquad
\alpha_{ij}=\frac{\exp(e_{ij}-m_i)}{\sum_t\exp(e_{it}-m_i)}.
$$

**Tự kiểm tra bằng đại số:** nhân tử $\exp(-m_i)$ xuất hiện ở cả tử và mẫu nên triệt tiêu; kết quả không đổi. Nếu các score là $(1000,1001)$, phiên bản trực tiếp dễ tràn số; sau khi trừ cực đại ta chỉ cần tính $(e^{-1},1)$.

Không gọi softmax trên toàn bộ cạnh của đồ thị. Cũng không chuẩn hóa theo chiều head: làm như vậy sẽ khiến các head cạnh tranh với nhau, thay vì các láng giềng cạnh tranh trong từng head.

## 6. Ví dụ tính tay: một bước GAT trên đồ thị ba nút

**Ví dụ do báo cáo tự xây dựng; số học đã kiểm tra bằng Python.** Chỉ xét cập nhật nút 0 trong đồ thị vô hướng có các cặp $\{0,1\}$, $\{0,2\}$ và thêm self-loop.

Đặt:

$$
h_0=(1,0)^T,\quad h_1=(0,1)^T,\quad h_2=(1,1)^T,\quad W=I_2,
$$

$$
a_{\mathrm{dst}}=(1,-1)^T,\qquad a_{\mathrm{src}}=(-1,2)^T.
$$

Vì $a=[a_{\mathrm{dst}}\Vert a_{\mathrm{src}}]$, ta có:

$$
a^T[g_i\Vert g_j]=a_{\mathrm{dst}}^Tg_i+a_{\mathrm{src}}^Tg_j.
$$

### 6.1. Tính score

Với nút nhận 0, $a_{\mathrm{dst}}^Tg_0=1$:

| Nút gửi $j$ | $g_j$ | $a_{\mathrm{src}}^Tg_j$ | Tổng trước LeakyReLU | $e_{0j}$ |
|---|---|---:|---:|---:|
| 0 | $(1,0)$ | −1 | 0 | 0 |
| 1 | $(0,1)$ | 2 | 3 | 3 |
| 2 | $(1,1)$ | 1 | 2 | 2 |

Ở đây các tổng không âm nên LeakyReLU giữ nguyên. Nếu một tổng bằng −1, LeakyReLU với slope 0,2 cho −0,2; không biến nó thành 0 như ReLU.

### 6.2. Chuẩn hóa

$$
Z=e^0+e^3+e^2\approx28.474593.
$$

$$
(\alpha_{00},\alpha_{01},\alpha_{02})
\approx(0.035119,\ 0.705385,\ 0.259496).
$$

Tổng các hệ số bằng 1 trong sai số làm tròn.

### 6.3. Tổng hợp

Chọn activation đồng nhất cho ví dụ, không dropout và không bias:

$$
h'_0=0.035119(1,0)+0.705385(0,1)+0.259496(1,1)
\approx(0.294615,\ 0.964881).
$$

Nếu chỉ lấy trung bình ba vector, đầu ra là $(2/3,2/3)$. Attention trong ví dụ tạo đầu ra khác vì nút 1 có hệ số lớn hơn. Điều này minh họa cơ chế tính, **không chứng minh** bộ tham số giả định đã học được điều hữu ích.

### 6.4. Các kết luận rút ra từ chính ví dụ

1. Self-loop không bảo đảm nút tự thân có trọng số lớn: ở đây hệ số của nút 0 nhỏ nhất.
2. Xóa một cạnh làm thay đổi mẫu số softmax, vì vậy có thể thay đổi hệ số của các cạnh còn lại.
3. $\alpha_{01}$ chỉ là trọng số khi nút 0 nhận tin từ nút 1; không phải xác suất có trích dẫn.
4. Đổi thứ tự liệt kê các cạnh không được làm thay đổi đầu ra toán học.
5. Muốn cập nhật nút 1 phải lập neighborhood và mẫu số riêng cho nút 1.

## 7. Multi-head, self-loop và dropout

### 7.1. Nhiều head và kích thước đầu ra

Gọi $u_i^{(k)}=\sum_j\alpha_{ij}^{(k)}W^{(k)}h_j$. Với $K$ head, tầng ẩn nối các đầu ra; tầng dự đoán có thể lấy trung bình trước activation cuối. [S1]

$$
h'_i=\mathop{\Vert}_{k=1}^K\sigma(u_i^{(k)})
\quad\text{hoặc}\quad
h'_i=\sigma\left(\frac1K\sum_{k=1}^Ku_i^{(k)}\right).
$$

Tài liệu `GATConv` phân biệt `heads`, `out_channels` và `concat`; `concat=False` lấy trung bình các head. [S9]

**Bảng suy ra kích thước để thiết kế layer của nhóm:**

| $N$ | $K$ | Số chiều mỗi head $F'$ | Chế độ | Shape đầu ra |
|---:|---:|---:|---|---|
| 2.708 | 1 | 8 | Concat | `[2708, 8]` |
| 2.708 | 8 | 8 | Concat | `[2708, 64]` |
| 2.708 | 8 | 8 | Average | `[2708, 8]` |
| 2.708 | 4 | 7 | Average | `[2708, 7]` |

Trong interface nên đặt tên `out_channels_per_head` hoặc ghi rõ `out_channels` là số chiều **mỗi head**. Kiểm thử phải phân biệt $F'$ và $KF'$, không áp dụng máy móc một shape cho mọi cấu hình.

### 7.2. Self-loop và cạnh trùng

Mã nguồn GAT của PyG xử lý self-loop bằng cách bỏ loop hiện có rồi thêm lại, và chia phép chiếu attention thành phần nguồn–đích trước khi cộng trên cạnh. [S10]

**Hệ quả thiết kế của nhóm:** utility phải bảo đảm mỗi nút có đúng một self-loop khi bật tùy chọn này. Hai bản sao của cùng một cạnh không chỉ lãng phí bộ nhớ: chúng có thể tham gia mẫu số hai lần, làm đổi tỷ lệ tổng đóng góp. Self-loop phục vụ message passing không được tự động biến thành positive label cho Link Prediction.

Một nút cô lập nhưng có self-loop vẫn nhận được vector của chính nó. Nếu bỏ loop, cần quy định rõ đầu ra khi không có thông điệp; không để hành vi này phụ thuộc tình cờ vào shape hoặc bias.

### 7.3. Dropout và điều kiện kiểm tra tổng attention

PyG định nghĩa tham số dropout của `GATConv` trên hệ số attention đã chuẩn hóa. [S9] Vì dropout có thể xóa và đổi tỷ lệ các hệ số, tổng sau dropout trong một lượt train không nhất thiết bằng 1.

Do đó, bài kiểm tra $\sum_j\alpha_{ij}=1$ nên dùng hệ số **trước dropout**, hoặc chạy chế độ evaluation. Feature dropout và attention dropout cần được đặt tên riêng trong cấu hình; chúng tác động lên hai đại lượng khác nhau.

## 8. Ánh xạ sang PyTorch Geometric

### 8.1. Chiều truyền thông điệp

Với `flow="source_to_target"`, `edge_index[0]` chứa nút gửi và `edge_index[1]` chứa nút nhận. `MessagePassing` mặc định có `node_dim=-2`. [S4]

Ví dụ tự tạo:

```python
edge_index = [[0, 1, 0, 2],
              [1, 0, 2, 0]]
```

Các cột lần lượt là $0\to1$, $1\to0$, $0\to2$, $2\to0$. Chưa có loop. Để tính nút 0, lấy các cột có hàng thứ hai bằng 0, không lấy theo hàng thứ nhất.

Sau khi reshape đặc trưng thành `[N, K, F_prime]`, chiều node là 0. Nếu giữ `node_dim=-2`, trục được chọn sẽ là trục head. Thiết kế multi-head này cần đặt rõ `node_dim=0`.

### 8.2. Bảng tensor dự kiến

Đặt $M$ là số **cột cạnh sau khi chuẩn hóa và thêm loop**, để tránh nhầm với số cặp vô hướng.

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| `x` | `[N, F]` | Đặc trưng đầu vào |
| `edge_index` | `[2, M]` | Nguồn và đích của thông điệp |
| `g` | `[N, K, F_prime]` | Kết quả chiếu |
| `g_j`, `g_i` | `[M, K, F_prime]` | Đặc trưng lấy theo đầu mút cạnh |
| `scores` | `[M, K]` | Score của từng cạnh, từng head |
| `alpha` | `[M, K]` | Softmax theo nhóm đích |
| `messages` | `[M, K, F_prime]` | `alpha[..., None] * g_j` |
| `aggregated` | `[N, K, F_prime]` | Tổng thông điệp theo đích |
| `out_concat` | `[N, K * F_prime]` | Đầu ra nối head |
| `out_average` | `[N, F_prime]` | Đầu ra trung bình head |

Bảng là suy diễn shape từ công thức, dùng làm hợp đồng thiết kế; chưa phải log chạy model.

### 8.3. Phân chia trách nhiệm hàm

PyG ánh xạ đối số có hậu tố `_j`, `_i` đến đặc trưng nguồn, đích khi truyền qua `propagate`. Có thể dùng tổng hợp `aggr="add"` sẵn có mà vẫn tự viết quy tắc thông điệp. [S3]

| Thành phần | Đề xuất cho implementation của B |
|---|---|
| `__init__` | Khai báo phép chiếu, tham số attention, head, dropout |
| `forward` | Kiểm tra input, chuẩn bị cạnh, chiếu đặc trưng, gọi truyền tin, ghép head |
| Hàm tính score | Tự viết score theo công thức GAT |
| Hàm neighborhood softmax | Tự hiện thực gom nhóm, trừ max, exp và chia tổng |
| `message` | Nhân hệ số với vector nút gửi |
| `aggregate` | Có thể dùng phép cộng của lớp cơ sở |

Kế hoạch yêu cầu tự viết softmax theo láng giềng. Vì vậy, đề xuất dùng thao tác scatter/gather cơ bản để hiện thực; dùng utility `softmax` của PyG làm **đối chiếu kết quả**, thay vì mặc định coi việc gọi utility đã thỏa yêu cầu tự cài. `GATConv` cũng chỉ dùng làm mốc kiểm thử, không làm implementation chính.

### 8.4. Chi phí tính toán — suy ra từ thiết kế

Với $K$ head và $F'$ chiều mỗi head:

- Chiếu toàn bộ nút: khoảng $O(NFKF')$.
- Tạo và cộng thông điệp: khoảng $O(MKF')$.
- Lưu attention: $O(MK)$; nếu materialize toàn bộ thông điệp, cần thêm $O(MKF')$.

Đây là đếm phép toán theo bảng tensor. Tránh tạo attention đặc `[N, N, K]` khi chỉ dùng các cạnh có sẵn. Tăng số head trong khi giữ $F'$ cố định cũng tăng tổng chiều ẩn, nên không được diễn giải mọi cải thiện là tác dụng riêng của “nhiều head”.

## 9. GAT trong hai pipeline của nhóm

### 9.1. Node Classification

Thiết kế khái niệm: $X$ và đồ thị đầu vào → encoder GAT → điểm lớp cho từng nút. Loss chỉ tính tại `train_mask`; validation chọn cấu hình và checkpoint; test chỉ đánh giá cuối theo kế hoạch.

Việc nút test hiện diện trong đồ thị không đồng nghĩa nhãn test được dùng để huấn luyện. Cần tách kiểm tra quyền truy cập đặc trưng/cấu trúc với quyền truy cập nhãn. Không biến một thiết lập transductive thành inductive chỉ bằng cách đổi tên biến.

### 9.2. Link Prediction

Theo kế hoạch nhóm:

$$
Z=f_\theta(X,E_{\mathrm{train}}),\qquad
s_{uv}=z_u^Tz_v,\qquad
\hat p_{uv}=\operatorname{sigmoid}(s_{uv}).
$$

Encoder tạo vector cho nút; decoder chấm điểm cặp nút. Vì $z_u^Tz_v=z_v^Tz_u$, decoder này phù hợp mục tiêu vô hướng đã chọn, không phân biệt ai trích dẫn ai.

| Giai đoạn | Cạnh được phép dùng để tính embedding | Tập dùng đánh giá |
|---|---|---|
| Phát triển và tuning | Train | Validation |
| Retrain cuối theo kế hoạch nhóm | Train + validation | Chưa dùng test để chọn cấu hình |
| Final test | Train + validation, với model đã retrain | Test |

Quy trình retrain là **lựa chọn của nhóm**, không phải yêu cầu bắt buộc của GAT. Cần chốt số epoch hoặc quy tắc dừng retrain từ giai đoạn phát triển, không chọn bằng test.

### 9.3. Rủi ro cần mang sang tuần 2

`RandomLinkSplit` hỗ trợ chia cạnh và tùy chọn `is_undirected=True` nhằm tránh rò rỉ cạnh đảo chiều giữa các phần chia. Nó cũng phân biệt cạnh cho message passing với cạnh làm nhãn giám sát. [S11]

Áp dụng kế hoạch nhóm, B cần kiểm tra:

1. Chia theo cặp vô hướng duy nhất; không để $(u,v)$ ở train và $(v,u)$ ở test.
2. Encoder trong tuning chỉ nhận train edges, kể cả lúc đánh giá validation.
3. Negative không trùng bất kỳ positive nào trong đồ thị gốc; các negative split không trùng nhau.
4. Full graph chỉ phục vụ xây dựng tập loại trừ khi lấy negative, không chuyển ngược vào encoder.
5. Tỷ lệ 85/5/10 áp dụng lên cạnh mục tiêu, không áp dụng nhầm lên node hoặc self-loop kỹ thuật.

Các mục này là yêu cầu thiết kế cần kiểm chứng ở tuần 2, chưa phải chứng nhận pipeline đã an toàn.

## 10. Giới hạn và cách diễn giải kết quả

### 10.1. GAT gốc có giới hạn static attention

Brody, Alon và Yahav chỉ ra rằng thứ hạng attention trong GAT gốc không phụ thuộc linh hoạt vào nút truy vấn; họ đề xuất GATv2 để khắc phục giới hạn này. [S12]

Với dạng $e_{ij}=\operatorname{LeakyReLU}(c_i+b_j)$ và slope dương, thêm cùng $c_i$ rồi qua hàm đơn điệu không đổi thứ tự các $b_j$. Vì vậy, “phụ thuộc đặc trưng” không đồng nghĩa “có thể đổi mọi thứ hạng láng giềng tùy nút nhận”. Giá trị hệ số vẫn có thể khác do neighborhood và chuẩn hóa khác nhau.

Nhóm vẫn triển khai GAT gốc theo đề bài; GATv2 chỉ là tài liệu để nhận diện giới hạn, không tự ý thay kiến trúc.

### 10.2. Các nguyên tắc diễn giải đề xuất

- Không coi attention lớn là bằng chứng quan hệ nhân quả hoặc bằng chứng một cạnh luôn quan trọng. Khi visualization, ghi rõ layer, head, node nhận và chế độ train/eval.
- Không kết luận GAT tốt hơn GCN khi hai model dùng split, seed hoặc ngân sách tuning khác nhau.
- Khi khảo sát số head, nên phân biệt giữ cố định chiều mỗi head và giữ cố định tổng chiều ẩn.
- Khi khảo sát độ sâu, kiểm tra cả hiệu năng và mức thay đổi embedding; không quy mọi suy giảm cho over-smoothing nếu chưa loại trừ lỗi tối ưu hoặc lỗi code.
- Tách kết quả paper, ví dụ tính tay và kết quả thực nghiệm của nhóm thành ba loại bằng chứng khác nhau.

Đây là các nguyên tắc phân tích do báo cáo đề xuất cho RQ2–RQ4; tuần 1 chưa có số đo để xác nhận các giả thuyết.

## 11. Checklist thiết kế và kiểm chứng cho các tuần sau

### 11.1. Tiêu chí kiểm thử GAT dự kiến

| Kiểm tra | Điều cần chứng minh | Lỗi có thể phát hiện |
|---|---|---|
| Ví dụ ba nút ở mục 6 | Đúng đầu ra với tham số cố định | Đảo nguồn–đích, sai score |
| Tổng attention trước dropout | Mỗi node nhận và mỗi head có tổng 1 | Sai nhóm softmax |
| Concat/average | Đúng hai loại shape | Nhầm tổng chiều với chiều mỗi head |
| Backward | Tham số có gradient, không NaN/Inf | Đứt computational graph |
| Cạnh đảo thứ tự | Đầu ra tương đương trong sai số số học | Phụ thuộc thứ tự ngoài ý muốn |
| Nút cô lập + self-loop | Có đầu ra đúng và hữu hạn | Bỏ mất nút hoặc sai `num_nodes` |
| Đối chiếu `GATConv` | Cùng weight, loop, bias, dropout, head và activation | Sai công thức triển khai |
| Sanity training | Có thể học trên tập nhỏ | Lỗi nối layer hoặc training loop |

Gradient hợp lệ không có nghĩa mọi phần tử gradient phải khác 0 trong mọi input. Một vài gradient có thể bằng 0 do cấu hình hoặc dữ liệu; cần tránh viết điều kiện kiểm thử sai.

### 11.2. Quyết định cần thống nhất với A

| Nội dung | Định hướng từ kế hoạch hoặc báo cáo |
|---|---|
| Dữ liệu bắt buộc | Cora trước; dataset khác là mở rộng |
| Đồ thị | Vô hướng, biểu diễn sparse |
| Self-loop | Thêm nhất quán, không trùng |
| Interface layer | `forward(x, edge_index)`; ghi rõ chiều mỗi head |
| Activation | Ghi rõ nằm trong layer hay ở network để tránh áp dụng hai lần |
| Trả attention | Nếu có, trả kèm đúng danh sách cạnh sau xử lý |
| Split và seed | Lưu để tái lập và dùng chung giữa model |
| Review | A review GAT; B review GCN |

### 11.3. Ánh xạ tài liệu vào repository dự kiến

| Đường dẫn đề xuất | Vai trò | Trạng thái trong yêu cầu hiện tại |
|---|---|---|
| `report/week01_member_b_gat.md` | Báo cáo nghiên cứu này | Đã biên soạn nội dung; chưa đưa vào repo |
| `README.md` | Problem Statement, RQ, convention | Nội dung nền có trong báo cáo; chưa tạo repo |
| `utils/edge_split.py` | Chia cạnh cho LP | Tuần 2 |
| `utils/graph_utils.py` | Loop, cạnh trùng, indexing | Tuần 4 |
| `models/gat_layer.py` | GAT tự cài | Tuần 6 |
| `tests/test_gat_layer.py` | Kiểm chứng thiết kế | Xây cùng tiến độ implementation |

## 12. Câu hỏi tự kiểm tra và đáp án ngắn

1. **GCN có học tham số không?** Có. $W$ được học; hệ số chuẩn hóa theo bậc không phải attention học được.
2. **$e_{ij}$ và $\alpha_{ij}$ khác nhau thế nào?** Score chưa chuẩn hóa và hệ số sau neighborhood softmax.
3. **Trong báo cáo, $i$ là nguồn hay đích?** Đích nhận thông điệp; $j$ là nguồn.
4. **Chuẩn hóa theo tập nào?** Các cạnh cùng đi vào một nút, riêng cho mỗi head.
5. **Tại sao không softmax toàn bộ cạnh?** Mỗi neighborhood cần một phân phối riêng.
6. **Self-loop để làm gì?** Cho nút một đường nhận thông tin từ chính nó.
7. **Tại sao cần tránh loop trùng?** Bản sao có thể làm thay đổi tổng đóng góp sau chuẩn hóa.
8. **8 head, mỗi head 8 chiều, concat có bao nhiêu chiều?** 64.
9. **Cùng cấu hình nhưng average thì sao?** 8.
10. **Tensor `[N,K,F_prime]` có node ở trục nào?** Trục 0.
11. **Tổng attention sau dropout có luôn bằng 1 không?** Không.
12. **Attention có phải xác suất tồn tại cạnh không?** Không; LP dùng decoder riêng.
13. **Có thể giữ cạnh validation trong encoder khi đánh giá validation LP không?** Không theo protocol nhóm.
14. **Hai chiều một cạnh vô hướng có được chia khác tập không?** Không.
15. **Ví dụ tính tay chứng minh điều gì?** Kiểm tra được phép tính trong cấu hình cụ thể; chưa chứng minh khả năng học.
16. **GAT có chắc tốt hơn GCN không?** Không; đó là câu hỏi cần thực nghiệm.
17. **Tự đọc hiểu báo cáo đã đồng nghĩa hoàn thành code tuần 1 chưa?** Không; repository và các việc phối hợp phải được thực hiện, kiểm tra riêng.

## 13. Kết quả nghiên cứu và bàn giao sang tuần 2

Báo cáo đã xác định được chuỗi tính toán của GAT, quy ước chiều cạnh, bảng tensor multi-head và ví dụ số có thể sử dụng làm mốc kiểm tra. Phần đối chiếu GCN giúp B hiểu nhiệm vụ review của mình; phần encoder–decoder và protocol giúp ngăn việc mang sai adjacency sang Link Prediction.

| Hạng mục | Kết quả của báo cáo |
|---|---|
| Tổng hợp tài liệu gốc và tài liệu chính thức | Có danh mục nguồn và vị trí sử dụng |
| Lý thuyết GAT | Có công thức, ký hiệu, phân tích |
| Ví dụ tính tay | Có, đã kiểm tra số học |
| Thiết kế tự cài | Có định hướng hàm, shape và test |
| Hiệu năng Cora | Chưa thực nghiệm |
| Pipeline dữ liệu, code GAT, repository | Chưa triển khai trong yêu cầu này |

Đầu việc ưu tiên ở tuần 2 là tải Cora, ghi thống kê thực tế, chuẩn hóa cạnh và xây split LP có kiểm thử. Những số liệu từ tài liệu trong báo cáo này là mốc đối chiếu, không thay thế kết quả quan sát trên môi trường của nhóm.

## 14. Tài liệu tham khảo và phạm vi sử dụng

**Ngày truy cập các nguồn trực tuyến: 27/09/2026.** Ưu tiên paper gốc, trang tác giả và tài liệu/mã nguồn chính thức. Các URL `latest` có thể thay đổi; khi thực hiện dự án cần khóa phiên bản package trong môi trường chạy.

**[P0] Kế hoạch nội bộ nhóm.** `Plan GNN.md`, do người dùng cung cấp. Dùng xác định vai trò B, RQ, lịch tuần, yêu cầu tự cài và protocol. Đây là đặc tả dự án, không phải bằng chứng thực nghiệm đã công bố.

**[S1] Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò, P.; Bengio, Y. (2018). _Graph Attention Networks_. ICLR 2018.** Bản arXiv đầu tiên năm 2017. Đọc mục 2.1 và bảng dữ liệu; dùng cho công thức gốc và multi-head.  
<https://arxiv.org/abs/1710.10903>  
Toàn văn: <https://arxiv.org/html/1710.10903v3>

**[S2] Kipf, T. N.; Welling, M. (2017). _Semi-Supervised Classification with Graph Convolutional Networks_. ICLR 2017.** Bản arXiv đầu tiên năm 2016. Đọc mục 2–3; dùng đối chiếu quy tắc chuẩn hóa và tham số học.  
<https://arxiv.org/abs/1609.02907>  
Toàn văn: <https://arxiv.org/html/1609.02907v4>

**[S3] PyTorch Geometric. _Creating Message Passing Networks_.** Dùng cho khung `propagate`, `message`, `aggregate`, `update` và quy ước `_i`, `_j`.  
<https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html>

**[S4] PyTorch Geometric. _MessagePassing API_.** Dùng kiểm tra chiều `source_to_target`, shape và `node_dim`.  
<https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html>

**[S5] PyTorch Geometric. _Planetoid dataset API_.** Dùng tham chiếu thống kê Cora và lựa chọn public split.  
<https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html>

**[S6] PyTorch Geometric. _Introduction by Example_.** Dùng cho đối tượng `Data` và biểu diễn `edge_index`.  
<https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html>

**[S7] Veličković và cộng sự. _Graph Attention Networks — Author Project Page_.** Dùng giải thích masked attention, chia sẻ tham số và động cơ tổng hợp cục bộ.  
<https://petar-v.com/GAT/>

**[S8] PyTorch Geometric. _Source code for torch_geometric.utils._softmax_.** Dùng tham khảo softmax theo nhóm và ổn định số.  
<https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/_softmax.html>

**[S9] PyTorch Geometric. _GATConv API_.** Dùng đối chiếu `heads`, `concat`, số chiều và attention dropout; không dùng lớp này thay implementation của nhóm.  
<https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html>

**[S10] PyTorch Geometric. _Source code for torch_geometric.nn.conv.gat_conv_.** Dùng kiểm tra phép tách attention nguồn–đích và xử lý self-loop.  
<https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html>

**[S11] PyTorch Geometric. _RandomLinkSplit API_.** Dùng tham khảo chia cạnh vô hướng và phân biệt cạnh cấu trúc với nhãn giám sát. Không coi việc gọi transform là bằng chứng mọi yêu cầu riêng của nhóm đã được thỏa mãn.  
<https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.RandomLinkSplit.html>

**[S12] Brody, S.; Alon, U.; Yahav, E. (2022). _How Attentive are Graph Attention Networks?_. ICLR 2022.** Bản arXiv đầu tiên năm 2021. Dùng bổ sung giới hạn static attention; không thay đổi phạm vi sang GATv2.  
<https://arxiv.org/abs/2105.14491>
