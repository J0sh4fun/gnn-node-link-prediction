# Từ Spectral Graph Convolution đến công thức GCN

**Bản tóm tắt và diễn giải bài báo _Semi-Supervised Classification with Graph Convolutional Networks_ — Thomas N. Kipf và Max Welling, ICLR 2017.**

Nguồn đối chiếu: arXiv:1609.02907v4, ngày 22/02/2017. Trọng tâm là phần 2, các phương trình (3)–(8), dẫn tới quy tắc lan truyền ở phương trình (2). Các giải thích toán học bổ sung được dùng để làm rõ những bước bài báo trình bày ngắn gọn.

> Ý chính: công thức GCN được xây dựng qua một chuỗi xấp xỉ, giới hạn họ bộ lọc và lựa chọn kiến trúc. Không phải mọi bước đều là biến đổi đại số tương đương; đặc biệt, renormalization là thay đổi toán tử truyền bá.

## 1. Bài toán và ký hiệu

### 1.1. Đồ thị và đặc trưng nút

Xét đồ thị vô hướng không có self-loop ban đầu, với trọng số cạnh không âm. Trường hợp đơn giản nhất là đồ thị không trọng số.

| Ký hiệu | Ý nghĩa | Kích thước |
|---|---|---|
| $N$ | Số nút | Vô hướng |
| $A$ | Ma trận kề đối xứng | $N\times N$ |
| $D$ | Ma trận bậc, $D_{ii}=d_i=\sum_j A_{ij}$ | $N\times N$ |
| $X$ | Ma trận đặc trưng nút, mỗi hàng ứng với một nút | $N\times C$ |
| $x$ | Tín hiệu một kênh trên đồ thị | $N$ |
| $I_N$ | Ma trận đơn vị | $N\times N$ |

Với đồ thị không trọng số, $A_{ij}=1$ nếu có cạnh nối $i$ và $j$, ngược lại bằng $0$.

Laplacian chưa chuẩn hóa và Laplacian chuẩn hóa đối xứng là:

$$
\Delta=D-A,\qquad L=I_N-D^{-1/2}AD^{-1/2}.
$$

Trong phần suy diễn, giả sử $d_i>0$. Nếu có nút cô lập, cần quy định cách xử lý, chẳng hạn đặt phần tử tương ứng của $D^{-1/2}$ bằng $0$ khi dùng biểu thức trên. Sau khi thêm self-loop ở bước renormalization, mọi bậc mới đều dương.

Do $L$ đối xứng thực, ta có phân rã:

$$
L=U\Lambda U^\top,\qquad U^\top U=I_N,
$$

trong đó $\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_N)$ và $0\le\lambda_i\le2$. Các vector riêng tạo thành cơ sở Fourier trên đồ thị; trị riêng nhỏ tương ứng với tín hiệu biến thiên ít theo cấu trúc đồ thị.

### 1.2. Phân loại nút bán giám sát

Chỉ một tập nhỏ nút có nhãn để huấn luyện. Mục tiêu là sử dụng cả đặc trưng $X$ và cấu trúc $A$ để dự đoán nhãn của những nút còn lại.

Một cách truyền thống là thêm Laplacian regularization, khuyến khích các nút nối nhau có đầu ra gần nhau. Với vector đầu ra $f$:

$$
\mathcal L_{\mathrm{reg}}=f^\top\Delta f
=\frac12\sum_{i,j}A_{ij}(f_i-f_j)^2.
$$

Với ma trận đầu ra $Q$:

$$
\mathcal L_{\mathrm{reg}}=\operatorname{tr}(Q^\top\Delta Q)
=\frac12\sum_{i,j}A_{ij}\|Q_i-Q_j\|^2.
$$

Hệ số $1/2$ xuất hiện vì tổng trên mọi cặp $(i,j)$ đếm mỗi cạnh vô hướng hai lần. Eq. (1) của bài báo lược bỏ hệ số này; trong hàm mất mát, nó có thể được hấp thụ vào hệ số điều chuẩn.

GCN đưa cấu trúc đồ thị trực tiếp vào mô hình $f(X,A)$ thay vì yêu cầu một số hạng điều chuẩn Laplacian riêng trong loss. Điều này không có nghĩa mô hình loại bỏ mọi thiên hướng làm trơn, hay không sử dụng điều chuẩn trọng số.

## 2. Tích chập phổ: điểm xuất phát

Theo cách tiếp cận spectral, biến đổi Fourier của tín hiệu $x$ là:

$$
\hat x=U^\top x.
$$

Một bộ lọc chéo trong miền phổ tác động lên các hệ số Fourier, sau đó biến đổi ngược:

$$
g_\theta\star x=U g_\theta U^\top x.
$$

Đây là Eq. (3) của bài báo. Khi tham số hóa bộ lọc theo trị riêng, ta viết $g_\theta(\Lambda)$.

Cách tính trực tiếp có hai khó khăn:

- Nhân với ma trận vector riêng dày $U$ tốn $O(N^2)$ cho một tín hiệu.
- Phân rã trị riêng đầy đủ của Laplacian rất tốn kém trên đồ thị lớn.

Miền phổ là cách tiếp cận được dùng để tạo động cơ cho GCN trong bài báo; không phải mọi định nghĩa convolution trên đồ thị đều bắt buộc đi qua Fourier.

## 3. Xấp xỉ Chebyshev: tránh phân rã trị riêng đầy đủ

### 3.1. Đổi thang phổ

Đặt:

$$
\tilde\Lambda=\frac{2}{\lambda_{\max}}\Lambda-I_N,
\qquad
\tilde L=\frac{2}{\lambda_{\max}}L-I_N.
$$

Với $\lambda_{\max}>0$ là trị riêng lớn nhất của $L$, phổ được đưa vào khoảng $[-1,1]$.

Các đa thức Chebyshev thỏa mãn:

$$
T_0(t)=1,\qquad T_1(t)=t,
$$

$$
T_k(t)=2tT_{k-1}(t)-T_{k-2}(t),\qquad k\ge2.
$$

Với đối số ma trận, $T_0(\tilde L)=I_N$.

### 3.2. Tham số hóa bộ lọc bằng đa thức

Xấp xỉ bộ lọc bởi đa thức bậc tối đa $K$:

$$
g_{\theta'}(\Lambda)\approx\sum_{k=0}^{K}\theta'_kT_k(\tilde\Lambda).
$$

Có $K+1$ hệ số: $\theta'_0,\ldots,\theta'_K$. Nhờ tính chất của đa thức ma trận:

$$
U T_k(\tilde\Lambda)U^\top=T_k(\tilde L),
$$

suy ra:

$$
g_{\theta'}\star x\approx\sum_{k=0}^{K}\theta'_kT_k(\tilde L)x.
$$

Đây là Eq. (4)–(5). Biểu thức không cần lưu hoặc nhân với $U$. Trong mô hình học, các hệ số Chebyshev được học trực tiếp; không cần xây dựng trước một bộ lọc phổ đầy đủ rồi mới xấp xỉ nó.

### 3.3. Tính cục bộ và chi phí tính toán

Một đa thức bậc $K$ theo Laplacian chỉ kết hợp thông tin từ các nút cách nút đang xét tối đa $K$ cạnh: tính chất **$K$-localized**.

Có thể tính lần lượt:

$$
v_0=x,\qquad v_1=\tilde Lx,\qquad
v_k=2\tilde Lv_{k-1}-v_{k-2},
$$

rồi lấy $\sum_{k=0}^{K}\theta'_kv_k$. Không cần tạo các lũy thừa ma trận dày.

Với biểu diễn thưa, chi phí cho một tín hiệu là $O(K(|E|+N))$. Cách viết $O(|E|)$ trong bài báo nhấn mạnh tính tuyến tính theo số cạnh khi giữ $K$ cố định và lược bỏ các chi phí liên quan đến nút. Vẫn cần biết hoặc ước lượng $\lambda_{\max}$ nếu dùng đúng phép đổi thang này; không cần toàn bộ phân rã trị riêng.

## 4. Giới hạn bậc nhất: chọn $K=1$

Khi $K=1$:

$$
g_{\theta'}\star x\approx\theta'_0x+\theta'_1\tilde Lx.
$$

Tiếp theo, bài báo dùng $\lambda_{\max}\approx2$. Do đó:

$$
\tilde L\approx L-I_N=-D^{-1/2}AD^{-1/2},
$$

và:

$$
g_{\theta'}\star x
\approx\theta'_0x-\theta'_1D^{-1/2}AD^{-1/2}x.
$$

Đây là Eq. (6), với hai tham số tự do. Bài báo kỳ vọng các tham số học được sẽ thích nghi với thay đổi thang phổ. Việc $\lambda_{\max}\le2$ không đồng nghĩa trị riêng lớn nhất luôn bằng $2$.

**Ý nghĩa của “first-order”:** bộ lọc là đa thức bậc nhất theo Laplacian. Đây không phải khai triển Taylor, và không có nghĩa toàn bộ lớp mạng sau khi thêm hàm kích hoạt là tuyến tính.

Một lớp chỉ truyền thông tin qua tối đa một cạnh. Xếp chồng $k$ lớp cho vùng tiếp nhận tối đa $k$-hop, nhưng mạng nhiều lớp có phi tuyến không tương đương nói chung với một bộ lọc Chebyshev bậc $k$.

Động cơ của tác giả là giảm độ phức tạp mỗi lớp và kỳ vọng hạn chế overfitting lên cấu trúc lân cận. Đây không phải bảo đảm rằng tăng độ sâu luôn cải thiện kết quả.

## 5. Ràng buộc hai tham số thành một

Đặt:

$$
\theta=\theta'_0=-\theta'_1.
$$

Thay vào biểu thức trên:

$$
g_\theta\star x
\approx\theta\left(I_N+D^{-1/2}AD^{-1/2}\right)x.
$$

Đây là Eq. (7). Tín hiệu của nút và tín hiệu tổng hợp từ láng giềng dùng chung hệ số học được.

Đây là **ràng buộc mô hình**, làm giảm số tham số và độ linh hoạt so với bộ lọc hai tham số. Với tín hiệu nhiều kênh, nó dẫn tới dùng chung một ma trận biến đổi cho thành phần bản thân và thành phần láng giềng.

## 6. Renormalization: thay toán tử truyền bá

### 6.1. Vì sao cần thay đổi?

Đặt $S=D^{-1/2}AD^{-1/2}$. Vì $L=I_N-S$, toán tử trước renormalization là:

$$
P=I_N+S=2I_N-L.
$$

Các trị riêng của $P$ nằm trong $[0,2]$. Khi lặp phép nhân với $P$, thành phần có trị riêng lớn hơn $1$ có thể tăng mạnh; thành phần có trị riêng nhỏ hơn $1$ có thể suy giảm.

Trong mạng sâu, điều này góp phần gây khó khăn về ổn định số và gradient.

### 6.2. Thêm self-loop rồi chuẩn hóa lại

Tác giả thay $P$ bằng:

$$
\tilde A=A+I_N,\qquad
\tilde D_{ii}=\sum_j\tilde A_{ij}=d_i+1,
$$

$$
\hat A=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}.
$$

Cần phân biệt rõ:

$$
I_N+D^{-1/2}AD^{-1/2}
\;\longrightarrow\;
\tilde D^{-1/2}(A+I_N)\tilde D^{-1/2}.
$$

Mũi tên biểu thị **lựa chọn thay toán tử**, không phải dấu bằng. Với đồ thị không self-loop ban đầu:

$$
\hat A_{ii}=\frac{1}{d_i+1},\qquad
\hat A_{ij}=\frac{A_{ij}}{\sqrt{(d_i+1)(d_j+1)}}\quad(i\ne j).
$$

Renormalization thay cả trọng số của bản thân nút và các láng giềng; nói chung không phải phép nhân toàn bộ toán tử cũ với một hằng số.

### 6.3. Ý nghĩa và giới hạn

Với đồ thị vô hướng, trọng số không âm, phổ của $\hat A$ nằm trong $[-1,1]$ và $\|\hat A\|_2\le1$. Riêng phép truyền bá vì thế không làm tăng chuẩn Euclid của tín hiệu.

Tuy nhiên, điều này không bảo đảm toàn bộ mạng tránh exploding/vanishing gradient: còn có ma trận trọng số, hàm kích hoạt và tác động của việc lặp nhiều lớp.

Chuẩn hóa đối xứng cũng không phải trung bình cộng thông thường: tổng mỗi hàng của $\hat A$ không nhất thiết bằng $1$.

## 7. Từ một kênh đến công thức GCN

Với $X\in\mathbb R^{N\times C}$ và $F$ kênh đầu ra, dùng ma trận tham số $\Theta\in\mathbb R^{C\times F}$:

$$
Z=\hat A X\Theta
=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}X\Theta,
\qquad Z\in\mathbb R^{N\times F}.
$$

Đây là Eq. (8). Ma trận $\hat A$ kết hợp thông tin giữa các nút, còn $\Theta$ trộn các kênh đặc trưng bằng tham số dùng chung trên toàn đồ thị.

Thêm hàm kích hoạt và trọng số riêng cho mỗi lớp:

$$
\boxed{
H^{(l+1)}=\sigma\!\left(\hat A H^{(l)}W^{(l)}\right),
\qquad H^{(0)}=X.
}
$$

Đây là quy tắc trung tâm, Eq. (2) của bài báo. Nếu $H^{(l)}$ có $C_l$ kênh thì $W^{(l)}\in\mathbb R^{C_l\times C_{l+1}}$.

### Chi phí của một lớp

Nếu $X$ dày, tính $B=X\Theta$ trước rồi tính $Z=\hat A B$ cho chi phí:

$$
O(NCF)+O((|E|+N)F).
$$

Nếu tổng hợp trước rồi biến đổi đặc trưng, chi phí tương ứng là $O((|E|+N)C+NCF)$. Đặc trưng thưa có thể giảm thêm chi phí.

Bài báo ghi $O(|E|FC)$ để nhấn mạnh khả năng mở rộng theo số cạnh. Khi phân tích triển khai, cần xét thứ tự nhân ma trận và số kênh.

Bộ nhớ lưu cấu trúc thưa là $O(|E|+N)$. Bộ nhớ huấn luyện còn bao gồm đặc trưng, kích hoạt trung gian, tham số, gradient và trạng thái bộ tối ưu.

## 8. Góc nhìn không gian và mối liên hệ với WL-1

### 8.1. Công thức cho từng nút

Với đồ thị không trọng số, đặt $\tilde d_i=d_i+1$. Quy tắc GCN tương đương:

$$
h_i^{(l+1)}=sigma\!\left(
\sum_{j\in\mathcal N(i)\cup\{i\}}
\frac{h_j^{(l)}W^{(l)}}{\sqrt{\tilde d_i\tilde d_j}}
\right).
$$

Mỗi nút nhận thông tin từ chính nó và láng giềng, kết hợp theo hệ số chuẩn hóa, rồi áp dụng phi tuyến. Với đồ thị có trọng số, hệ số tổng hợp có thêm tử số $\tilde A_{ij}$.

Self-loop giúp giữ lại đặc trưng của chính nút trong phép tổng hợp. Bậc trong mẫu số phải là bậc **sau khi thêm self-loop**.

### 8.2. Liên hệ với Weisfeiler–Lehman

Phụ lục A đưa ra một cách nhìn trực giác: thay thao tác cập nhật nhãn rời rạc bằng phép biến đổi khả vi có tham số.

Để mô tả WL-1 chặt chẽ, có thể viết:

$$
c_i^{(t+1)}=\operatorname{HASH}\!\left(
 c_i^{(t)},\{\!\{c_j^{(t)}:j\in\mathcal N(i)\}\!\}
\right).
$$

Ký hiệu $\{\!\{\cdot\}\!\}$ là đa tập, giữ cả giá trị và số lần xuất hiện. Hàm mã hóa cần phân biệt các đầu vào khác nhau. Cách viết hash của tổng màu trong Phụ lục A là mô tả giản lược; cộng mã màu thông thường có thể mất thông tin, chẳng hạn $1+3=2+2$.

GCN và WL-1 cùng cập nhật biểu diễn qua lân cận, nhưng không nên suy ra GCN có sức phân biệt tương đương WL-1. Nghiên cứu bổ sung của Xu et al. (2019) phân tích giới hạn biểu diễn của GCN và các bộ tổng hợp lân cận; kết quả đó không thuộc bài báo năm 2017.

Phụ lục A.1 minh họa một GCN ba lớp với trọng số ngẫu nhiên, $X=I_N$, trên mạng câu lạc bộ karate. Ví dụ cho thấy cấu trúc truyền bá có thể tạo embedding hữu ích trước huấn luyện; đây không phải bảo đảm cho mọi đồ thị hoặc mọi khởi tạo.

## 9. GCN hai lớp cho phân loại bán giám sát

Tính trước $\hat A$ từ đồ thị. Mô hình hai lớp là:

$$
Z=\operatorname{softmax}\!\left(
\hat A\operatorname{ReLU}(\hat AXW^{(0)})W^{(1)}
\right).
$$

Trong đó:

- $W^{(0)}\in\mathbb R^{C\times H}$ biến đổi sang $H$ kênh ẩn.
- $W^{(1)}\in\mathbb R^{H\times F}$ biến đổi sang $F$ lớp nhãn.
- Softmax áp dụng theo từng hàng, cho phân phối xác suất lớp của mỗi nút.

Với tập nút có nhãn dùng để huấn luyện $\mathcal Y_L$ và nhãn one-hot $Y$:

$$
\mathcal L_{\mathrm{sup}}
=-\sum_{i\in\mathcal Y_L}\sum_{f=1}^{F}Y_{if}\log Z_{if}.
$$

**Nút không có nhãn vẫn tham gia truyền bá đặc trưng.** Chỉ các nút thuộc $\mathcal Y_L$ đóng góp trực tiếp vào supervised loss. Trong thiết lập transductive, cấu trúc và đặc trưng của các nút cần dự đoán đã hiện diện trong đồ thị khi huấn luyện, nhưng nhãn kiểm tra không được dùng để tối ưu mô hình.

Bài báo huấn luyện full-batch bằng Adam, có dropout, L2 regularization trên trọng số lớp đầu và early stopping dựa trên validation. Vì vậy, không dùng Laplacian regularization riêng không có nghĩa là không dùng điều chuẩn.

## 10. Kết quả thực nghiệm và phạm vi kết luận

### 10.1. Accuracy

Các số dưới đây là accuracy (%), từ Bảng 2 của bài báo, trên cách chia dữ liệu chuẩn được sử dụng trong thí nghiệm:

| Phương pháp | Citeseer | Cora | Pubmed | NELL |
|---|---:|---:|---:|---:|
| ManiReg | 60.1 | 59.5 | 70.7 | 21.8 |
| SemiEmb | 59.6 | 59.0 | 71.1 | 26.7 |
| LP | 45.3 | 68.0 | 63.0 | 26.5 |
| DeepWalk | 43.2 | 67.2 | 65.3 | 58.1 |
| ICA | 69.1 | 75.1 | 73.9 | 23.1 |
| Planetoid* | 64.7 | 75.7 | 77.2 | 61.9 |
| **GCN** | **70.3** | **81.5** | **79.0** | **66.0** |

Planetoid* là biến thể Planetoid tốt nhất được chọn cho từng bộ dữ liệu. GCN có accuracy cao nhất trong bảng này. Kết luận gắn với thiết lập thực nghiệm; bài báo cũng báo cáo kết quả trên các cách chia ngẫu nhiên và cho thấy hiệu năng thay đổi theo cách chia dữ liệu.

### 10.2. Thời gian huấn luyện

Bảng 2 đối chiếu thời gian đến hội tụ của GCN và Planetoid trên cùng phần cứng:

| Bộ dữ liệu | GCN | Planetoid |
|---|---:|---:|
| Citeseer | 7 giây | 26 giây |
| Cora | 4 giây | 13 giây |
| Pubmed | 38 giây | 25 giây |
| NELL | 48 giây | 185 giây |

GCN nhanh hơn trên ba bộ dữ liệu, nhưng chậm hơn trên Pubmed. Không có cơ sở từ bảng này để kết luận GCN nhanh hơn mọi baseline trên mọi dữ liệu.

### 10.3. Renormalization và độ sâu

Trong Bảng 3, biến thể renormalization đạt accuracy cao nhất trong các mô hình truyền bá được so sánh trên cả Citeseer, Cora và Pubmed. Đây là bằng chứng thực nghiệm cho lựa chọn kiến trúc trong phạm vi các cấu hình được thử.

Phụ lục B báo cáo kết quả tốt nhất với hai hoặc ba lớp trong các thí nghiệm về độ sâu. Tăng vùng tiếp nhận bằng cách thêm lớp không bảo đảm tăng accuracy.

## 11. Bảng tổng hợp chuỗi suy diễn

| Bước | Biểu thức hoặc thao tác | Bản chất |
|---|---|---|
| 1 | $Ug_\theta(\Lambda)U^\top x$ | Định nghĩa tích chập phổ |
| 2 | $\sum_{k=0}^{K}\theta'_kT_k(\tilde L)x$ | Xấp xỉ/tham số hóa đa thức cục bộ |
| 3 | Chọn $K=1$ | Giới hạn bộ lọc ở bậc nhất |
| 4 | Đặt $\lambda_{\max}\approx2$ | Đơn giản hóa thang phổ |
| 5 | $\theta'_0=\theta$, $\theta'_1=-\theta$ | Ràng buộc tham số |
| 6 | $I_N+D^{-1/2}AD^{-1/2}\to\hat A$ | Thay toán tử bằng renormalization |
| 7 | $Z=\hat AX\Theta$ | Mở rộng sang nhiều kênh |
| 8 | $H^{(l+1)}=\sigma(\hat AH^{(l)}W^{(l)})$ | Xây dựng lớp GCN |

Công thức cuối cùng kết hợp ba thao tác: truyền bá theo đồ thị đã thêm self-loop và chuẩn hóa, biến đổi các kênh bằng trọng số học được, rồi áp dụng phi tuyến.

## Tài liệu tham khảo

1. Kipf, T. N., & Welling, M. (2017). _Semi-Supervised Classification with Graph Convolutional Networks_. ICLR 2017. [arXiv:1609.02907v4](https://arxiv.org/abs/1609.02907v4). Nguồn chính: phần 2–3, Bảng 2–3, Phụ lục A–B.
2. Hammond, D. K., Vandergheynst, P., & Gribonval, R. (2011). _Wavelets on graphs via spectral graph theory_. Applied and Computational Harmonic Analysis, 30(2), 129–150. Nguồn nền tảng được bài báo dẫn cho cách tiếp cận phổ và Chebyshev.
3. Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). _Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering_. NeurIPS 2016. Nguồn nền tảng được bài báo dẫn cho ChebNet.
4. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). _How Powerful are Graph Neural Networks?_ ICLR 2019. [arXiv:1810.00826](https://arxiv.org/abs/1810.00826). Nguồn bổ sung cho lưu ý về sức biểu diễn và WL-1, không thuộc suy diễn gốc của Kipf & Welling.
