# Tuần 4 — Kiến trúc framework huấn luyện và GCN sparse

**Phụ trách:** Thành viên A. **Phạm vi:** hạ tầng node classification đã kiểm chứng trong Tuần 4 và kế hoạch triển khai GCN tự cài đặt ở Tuần 5.

Tài liệu mô tả [trainer](../../train/trainer.py), [các phép toán đồ thị](../../utils/graph_ops.py), [kiểm thử graph operations](../../tests/test_graph_ops.py) và [API đánh giá](../../utils/evaluate.py) hiện có. Bản bàn giao Tuần 4 ghi nhận kiểm chứng thành công, độ bao phủ kiểm thử graph operations đạt 100% và gradient đã được xác minh. Đây là trạng thái do nhóm cung cấp, không phải kết quả đo coverage mới; một tỷ lệ coverage chưa xác định loại đo cũng không đồng nghĩa đã đạt đồng thời statement coverage và branch coverage. Lộ trình dưới đây là công việc dự kiến, không phải một mô hình GCN đã hoàn thành.

## 1. Kiến trúc huấn luyện mô-đun

### Phân tách trách nhiệm

`NodeClassificationTrainer` nhận mô hình, optimizer, hàm cross-entropy, giới hạn epoch, patience, đường dẫn đầu ra và `run_config` có thể tuần tự hóa thành JSON. Kiến trúc mô hình xác định cách tính logits; trainer quản lý tối ưu, lựa chọn theo validation, khôi phục checkpoint và báo cáo cuối cùng.

| Thành phần | Trách nhiệm | Ranh giới |
|---|---|---|
| `nn.Module` | Biến đổi đầu vào nút thành logits thô `[N, C]` | Không lựa chọn split hoặc tính metric trong mô hình |
| `get_cora_data` | Đặc trưng, nhãn, public masks và liên kết vô hướng | Cung cấp dữ liệu, không chọn checkpoint |
| `NodeClassificationTrainer` | Vòng đời huấn luyện và lựa chọn theo validation loss | Nhận mô hình cùng optimizer đã được cấu hình |
| `utils/graph_ops.py` | Self-loop, hệ số đối xứng và tổng hợp sparse | Không nhận nhãn, mask, optimizer hoặc tham số học được |
| `utils/evaluate.py` | Metric tương ứng từng bài toán | Không tạo split hoặc sửa adjacency bị leakage |

`_forward` gọi `model(x)` nếu `edge_index=None` và `model(x, edge_index)` nếu có tensor cạnh. Hàm không bắt `TypeError` từ bên trong mô hình rồi âm thầm thử một chữ ký khác. Vì vậy, cùng vòng đời tối ưu có thể phục vụ baseline MLP, GCN tự cài đặt và GAT trong giai đoạn sau.

`EpochResult` lưu training loss, validation loss, accuracy và macro-F1 của từng epoch. `TrainingResult` bất biến lưu epoch được chọn, epoch dừng, validation loss tốt nhất, các metric validation tương ứng, trạng thái dừng và toàn bộ lịch sử epoch. Epoch dừng cuối cùng và checkpoint được chọn là hai khái niệm khác nhau.

### Thực thi protocol thí nghiệm

Hàm mục tiêu huấn luyện là

$$
\mathcal L_{\mathrm{train}}
=\frac{1}{|\mathcal V_{\mathrm{train}}|}
\sum_{i\in\mathcal V_{\mathrm{train}}}
\operatorname{CE}(z_i,y_i).
$$

`fit(x, edge_index, y, train_mask, val_mask)` không nhận `test_mask` và không tham chiếu biến này. Hàm từ chối mask train/validation rỗng, sai cấu trúc hoặc giao nhau. Mỗi epoch tối ưu ở chế độ training, sau đó đánh giá validation với `eval()` trong `torch.no_grad()`. Chỉ khi **validation loss giảm nghiêm ngặt**, checkpoint được chọn mới được thay thế; loss bằng nhau vẫn làm tăng số epoch không cải thiện. Cấu hình mặc định là tối đa 200 epoch, patience 20.

Trạng thái được chọn được chụp bằng `deepcopy(model.state_dict())`, bao gồm tham số và registered buffers. Trainer khôi phục trạng thái này khi kết thúc. `evaluate_test(...)` khôi phục lại một lần nữa trước khi tính metric cuối cùng, từ chối mask giao với các nút train/validation đã dùng và chỉ cho phép một lần đánh giá cuối thành công trong mỗi lần fit. Nếu ghi log cuối thất bại, lần thử lại công bố metric đã lưu trong bộ nhớ thay vì tính lại.

Các cơ chế trên thực thi protocol không rò rỉ nhãn giữ lại trong phạm vi trainer. Chúng không kiểm tra được toàn bộ quy trình xây dựng đặc trưng/cạnh hoặc ngăn người thực nghiệm tạo lần chạy mới rồi lựa chọn theo điểm test. Trong node classification transductive, truy cập toàn bộ đồ thị được cho phép; nhãn giữ lại vẫn bị loại khỏi supervised loss. Lần gọi `fit` tiếp theo đặt lại lịch sử lựa chọn nhưng **không khởi tạo lại trọng số mô hình hoặc trạng thái optimizer**; thí nghiệm độc lập cần các instance mô hình/optimizer mới.

### CPU deterministic và lưu trạng thái

**Determinism thuộc trách nhiệm của script gọi, không phải tính năng tự động của trainer.** Trước khi tạo mô hình, entry point phải seed Python `random`, NumPy và PyTorch, đưa mô hình cùng tensor lên CPU, cố định số luồng CPU và bật `torch.use_deterministic_algorithms(True)`. Hàm `set_seed` hiện có trong [`train/train_mlp.py`](../../train/train_mlp.py) thực hiện cấu hình này. Cần ghi nhận cùng seed, split, tiền xử lý, phiên bản thư viện và cấu hình phần cứng. PyTorch không bảo đảm kết quả đồng nhất trên mọi nền tảng hoặc phiên bản. [Hướng dẫn tái lập kết quả của PyTorch](https://docs.pytorch.org/docs/2.7/notes/randomness.html)

Cả `checkpoint_dir / "best_model.pt"` và JSON log đều dùng `NamedTemporaryFile` trong thư mục đích, đóng file rồi thực hiện `Path.replace`. Thư mục cha được tạo khi cần; file tạm được dọn khi có lỗi. Đường dẫn được phân giải khi khởi tạo trainer; cấu hình không được ghi đè các trường kết quả dành riêng.

Log lần lượt có trạng thái `running`, `fitted` và `evaluated`; metric test chỉ xuất hiện ở giai đoạn cuối. Từng thao tác thay file có tính nguyên tử theo cơ chế của hệ thống file. Checkpoint và JSON **không phải một giao dịch nguyên tử chung**, và triển khai chưa cung cấp bảo đảm bền vững khi mất điện dựa trên `fsync`. Trạng thái tốt nhất trong bộ nhớ là nguồn chính thức cho đánh giá cuối; state dictionary xuất ra chưa phải checkpoint đầy đủ gồm optimizer/RNG để tiếp tục huấn luyện sau khi khởi động lại tiến trình.

## 2. Cơ sở toán học và độ phức tạp sparse

### Từ bộ lọc phổ bậc nhất đến toán tử đã triển khai

Với ma trận kề đối xứng, không âm $A$, xét Laplacian chuẩn hóa trên không gian các nút có bậc dương:

$$
L=I-D^{-1/2}AD^{-1/2}=U\Lambda U^\top.
$$

Bộ lọc phổ tác động theo $g_\theta(L)h=Ug_\theta(\Lambda)U^\top h$. Phép xấp xỉ bậc nhất và tái chuẩn hóa dựa trên [Kipf và Welling, Mục 2.1–2.2](https://arxiv.org/pdf/1609.02907#page=2). Cụ thể:

$$
g_\theta(L)h
\approx \theta_0h+\theta_1
\left(\frac{2}{\lambda_{\max}}L-I\right)h
\approx \theta_0h-\theta_1D^{-1/2}AD^{-1/2}h.
$$

Xấp xỉ thứ hai sử dụng $\lambda_{\max}\approx2$. Ràng buộc $\theta_0=-\theta_1=\theta$ cho $\theta(I+D^{-1/2}AD^{-1/2})h$. Bước tái chuẩn hóa tiếp theo là **thay thế toán tử**, không phải đẳng thức đại số:

$$
I+D^{-1/2}AD^{-1/2}
\ \longmapsto\
\hat A=\tilde D^{-1/2}\tilde A\tilde D^{-1/2},
\qquad
\tilde A=A+I,\quad
\tilde d_i=\sum_j\tilde A_{ij}.
$$

Sau khi đưa vào phép biến đổi giữa các kênh đặc trưng:

$$
H^{(l+1)}=\sigma\!\left(\hat A H^{(l)}W^{(l)}\right).
$$

Với `edge_index[:, e] = [j, i]`, cạnh truyền thông tin từ nút nguồn $j$ tới nút đích $i$, tương ứng vị trí ma trận $(i,j)$:

$$
\hat a_e=\frac{\tilde w_{j\to i}}{\sqrt{\tilde d_i\tilde d_j}},
\qquad
Z_i=\sum_{j\to i}\hat a_e(H_jW).
$$

| Bước toán học | Triển khai |
|---|---|
| Thêm liên kết với chính nút | `add_remaining_self_loops` tạo một loop mỗi nút cho input không trọng số và khởi tạo toàn bộ trọng số đầu ra |
| Bảo toàn trọng số được cung cấp | `compute_symmetric_norm` giữ weighted loop hiện hữu, thêm loop còn thiếu có trọng số một và cộng weighted edges trùng |
| Tính bậc sau bổ sung loop | `degree.index_add_(0, target, edge_weight)` sau khi thêm loop và coalesce |
| Tính hệ số | `inv_deg[target] * edge_weight * inv_deg[source]` |
| Áp dụng $\hat A$ | `sparse_spmm` đổi source/target sang row/column COO, coalesce rồi gọi `torch.sparse.mm` |

Với Cora chưa có loop, đây chính xác là $\tilde A=A+I$. Với input có trọng số đã bổ sung loop, hàm **chỉ thêm loop còn thiếu**, giữ nguyên giá trị đường chéo hiện hữu, kể cả bằng không. Chuẩn hóa lại các trọng số đã chuẩn hóa không phải phép toán lũy đẳng. Tính vô hướng và tổng trọng số bằng nhau ở hai chiều là tiền điều kiện do bên gọi bảo đảm; helper không tự đối xứng hóa hoặc xác minh điều kiện đối xứng đó.

Với $Y=\hat AX$, đạo hàm ngược cho:

$$
\frac{\partial\mathcal L}{\partial X}
=\hat A^\top\frac{\partial\mathcal L}{\partial Y}.
$$

Trên đồ thị vô hướng, $\hat A^\top=\hat A$. Triển khai giữ đường tính toán này kết nối với PyTorch autograd, cho phép gradient đi qua $X=HW$ tới $W$. Kết quả xác minh gradient theo input không nên được mở rộng thành một bảo đảm chưa kiểm thử về học trọng số cạnh tại các điểm kỳ dị có bậc bằng không.

### Chi phí tính toán và bộ nhớ sparse

Gọi $m$ là số liên kết vô hướng khác nhau, không kể loop; $M=2m+N$ là số phần tử được lưu sau khi bổ sung loop. Tương đương, $M=|\tilde{\mathcal E}|$ nếu tập cạnh đếm cả hai chiều và self-loop.

| Phép toán hoặc vùng lưu trữ | Biểu diễn sparse | Ma trận kề dense |
|---|---:|---:|
| Tổng hợp $F$ kênh đặc trưng | $\mathcal O(MF)$ | $\mathcal O(N^2F)$ |
| Lưu ma trận kề | $\mathcal O(M)$ | $\mathcal O(N^2)$ |
| Lưu đặc trưng/đầu ra nút | $\mathcal O(NF)$ | $\mathcal O(NF)$ |
| Tính bậc | $\mathcal O(M+N)$ | $\mathcal O(N^2)$ nếu quét mọi phần tử |

Do đó, phát biểu quen thuộc $\mathcal O(|\mathcal E|F)$ cần hiểu theo số phần tử sparse được lưu, bao gồm self-loop; với đồ thị không có cạnh gốc, không được bỏ qua thành phần $\mathcal O(NF)$ của các loop. Coalesce có thể phát sinh chi phí sắp xếp, thường tới $\mathcal O(M\log M)$. Helper hiện tạo/coalesce tensor COO khi gọi, nên thời gian thực thi tổng thể không chỉ gồm số phép tính SpMM.

Chiếu đặc trưng trước khi tổng hợp cho chi phí mỗi tầng:

$$
\mathcal O(NF_{\mathrm{in}}F_{\mathrm{out}})
+\mathcal O(MF_{\mathrm{out}}),
$$

cộng thêm chuẩn hóa/coalesce. Bộ nhớ tham số và activation vẫn cần thiết dù adjacency là sparse.

[Đặc tả Cora của PyG](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Planetoid.html) cho $N=2708$ và 10.556 phần tử cạnh được lưu. Sau khi thêm unit loop, $M=13.264$. Một ma trận kề dense float32 chứa 7.333.264 phần tử, xấp xỉ **27,97 MiB**. COO tối thiểu với hai chỉ số int64 và một trọng số float32 cho mỗi phần tử chỉ cần khoảng **0,253 MiB**, chưa tính tensor tạm và overhead của framework. Phép tổng hợp dense xét số vị trí ma trận nhiều hơn khoảng 553 lần; đây không phải cam kết về hệ số tăng tốc thời gian thực.

Ma trận kề dense của Cora có thể vừa bộ nhớ trên phần cứng thông thường. Việc cấm dense là **ràng buộc kiến trúc nhằm bảo đảm khả năng mở rộng** trong kế hoạch: tránh công việc bậc hai, các bản sao adjacency không cần thiết và một cách triển khai không phù hợp với đồ thị sparse lớn hơn. Đặc trưng nút dense $N\times F$ và phép chiếu đặc trưng học được vẫn được phép sử dụng.

### Bán xác định dương và cận phổ nghiêm ngặt

**Đối tượng bán xác định dương là Laplacian chuẩn hóa**
$\tilde L=I-\hat A$, **không nhất thiết là $\hat A$**. Giả sử đồ thị hữu hạn, không rỗng, $\tilde A=\tilde A^\top$ có phần tử không âm và mỗi nút có self-loop dương nghiêm ngặt $s_i=\tilde A_{ii}>0$. Bậc được tính từ chính adjacency đã bổ sung loop, nên $\tilde d_i>0$.

Với mọi $z\in\mathbb R^N$, đặt $u=\tilde D^{-1/2}z$. Khi đó:

$$
z^\top\tilde Lz
=u^\top(\tilde D-\tilde A)u
=\frac12\sum_{i,j}\tilde A_{ij}(u_i-u_j)^2
\ge0.
$$

Suy ra $\tilde L\succeq0$. Nếu $S$ là ma trận đường chéo chứa trọng số loop được thêm, thì $(D+S)-(A+S)=D-A$: thêm loop bảo toàn tính bán xác định dương của Laplacian chưa chuẩn hóa; bước chuẩn hóa là phép biến đổi đồng dư bằng ma trận đường chéo dương khả nghịch.

Đối với cận trên:

$$
z^\top(2I-\tilde L)z
=u^\top(\tilde D+\tilde A)u
=\frac12\sum_{i,j}\tilde A_{ij}(u_i+u_j)^2
\ge2\sum_i s_i u_i^2.
$$

Đặt $\alpha=\min_i(s_i/\tilde d_i)>0$. Do đó:

$$
z^\top(2I-\tilde L)z\ge2\alpha\|z\|_2^2,
\qquad
0\le\lambda(\tilde L)\le2-2\alpha<2.
$$

Như vậy, $\operatorname{spec}(\tilde L)\subset[0,2)$ mà không cần giả thiết đồ thị liên thông. Phổ tương ứng của $\hat A$ nằm trong $(-1,1]$ và có thể chứa trị riêng âm. Phản ví dụ: hai nút nối với nhau bởi cạnh trọng số hai, mỗi nút có unit loop:

$$
\hat A=\frac13\begin{bmatrix}1&2\\2&1\end{bmatrix},
\qquad
\operatorname{spec}(\hat A)=\{1,-1/3\}.
$$

**Vì sao phải thêm loop trước khi tính bậc:** các đẳng thức dạng toàn phương yêu cầu $\tilde D_{ii}=\sum_j\tilde A_{ij}$. Với hai nút nối bởi cạnh trọng số một, nếu dùng $D=I$ cũ sau khi thêm unit loop, ta thu được $I-(A+I)=-A$, có trị riêng $-1$. Bảo đảm bán xác định dương không còn đúng.

**Giới hạn của định lý:** mã nguồn cho phép weighted loop được khai báo rõ với trọng số bằng không. Input này không thỏa điều kiện mọi loop dương nghiêm ngặt; đồ thị hai phía có loop bằng không có thể vẫn giữ trị riêng Laplacian bằng 2. Cơ chế thay nghịch đảo bậc vô hạn bằng không bảo đảm hệ số forward hữu hạn ở nút bậc không, nhưng không chứng minh được cận phổ nghiêm ngặt. Cora chuẩn chưa có loop rồi được thêm unit loop thỏa định lý; nút cô lập ban đầu trở thành một liên kết với chính nó có trọng số một, với Laplacian chuẩn hóa bằng không tại nút đó. Các bảo đảm phổ chỉ áp dụng cho phép lan truyền đồ thị tuyến tính, không phải tính ổn định tổng thể của mọi trọng số học được, phi tuyến hoặc mạng nhiều tầng.

## 3. Hợp đồng giao tiếp và bàn giao với Thành viên B

### Phân công module đánh giá dùng chung

| Phụ trách | Hàm và đầu vào | Đầu ra và nghĩa vụ |
|---|---|---|
| Thành viên A | `evaluate_node_classification(logits, targets, mask)`; `[N,C]`, `[N]`, boolean `[N]` | `{"accuracy": float, "macro_f1": float}`; argmax, mask đồng bộ, detach và chuyển CPU; macro averaging với `zero_division=0` |
| Thành viên B | `evaluate_link_prediction(pos_pred, neg_pred)`; các vector điểm cạnh liên tục, một chiều, không rỗng | `{"roc_auc": float, "average_precision": float}`; tạo nhãn nhị phân, tính ROC-AUC/AP không đặt ngưỡng |

Theo mặc định scikit-learn hiện được sử dụng, bộ đánh giá nút không tính vào trung bình các lớp vắng ở cả nhãn thật lẫn dự đoán sau mask. Hai thành viên phải giữ cùng quy ước khi so sánh. Hàm link prediction trong checkout được rà soát vẫn là stub khi Thành viên B đang hoàn thiện Phiên 3–4 của Tuần 4. Khi triển khai, thay test `NotImplementedError` bằng các test metric, đồng thời giữ toàn bộ regression test của node classification.

### Hai protocol transductive với đối tượng giữ lại khác nhau

| Khía cạnh | Thành viên A: node classification | Thành viên B: link prediction |
|---|---|---|
| Đối tượng giữ lại | Nhãn nút theo public masks cố định | Các cặp nút positive và negative |
| Nút/đặc trưng được biết | Toàn bộ tập nút và đặc trưng Cora | Toàn bộ tập nút và đặc trưng |
| Adjacency khi tuning | Toàn bộ đồ thị trích dẫn vô hướng | **Chỉ positive train edges** đã đối xứng hóa |
| Sử dụng validation | Chọn checkpoint/siêu tham số bằng nhãn nút validation | Chọn bằng cạnh validation giữ lại, không đưa chúng vào adjacency khi tuning |
| Đánh giá cuối | Khôi phục checkpoint tốt nhất theo validation, đánh giá test node mask một lần | Theo Mục 3.3: retrain từ đầu trên train + validation positives, rồi đánh giá cạnh test |
| Metric dùng chung | Accuracy và Macro-F1 | ROC-AUC và Average Precision |

Sử dụng toàn bộ đồ thị là hợp lệ trong benchmark transductive giữ lại nhãn nút, vì liên kết là thông tin quan sát được còn nhãn là mục tiêu đánh giá. Trong link prediction, chính cạnh là mục tiêu; đưa cạnh giữ lại vào message passing sẽ làm lộ đáp án. `NodeClassificationTrainer` không triển khai edge loss hoặc kiểm soát edge split và không được dùng nguyên trạng như trainer cho link prediction.

Đối với Thành viên B, chia theo cặp không thứ tự để hai chiều không rơi vào các tập khác nhau. Negative phải vắng trong toàn bộ đồ thị positive gốc, không trùng giữa các split và tuân thủ tỷ lệ 1:1 trong kế hoạch. Thông tin toàn đồ thị chỉ được dùng để loại positive khỏi negative sampling, không được dùng làm adjacency của encoder. Thêm self-loop và tính bậc **từ adjacency được phép của từng giai đoạn**; không tính bậc trên toàn đồ thị rồi chỉ xóa hệ số của cạnh giữ lại. Loại các cặp một nút với chính nó khỏi target cạnh trích dẫn. Hủy cache chuẩn hóa khi tập cạnh được phép thay đổi.

## 4. Lộ trình Tuần 5: triển khai GCN tự cài đặt

### Phiên 1 — Triển khai và kiểm chứng `GCNLayer`

Tạo `models/gcn.py` với `GCNLayer(nn.Module)`. Tầng sở hữu ma trận học được $W\in\mathbb R^{F_{\mathrm{in}}\times F_{\mathrm{out}}}$, bias tùy chọn, khởi tạo Xavier/Glorot uniform và bias bằng không.

Giao diện dự kiến:

~~~python
GCNLayer(in_channels: int, out_channels: int, bias: bool = True)
forward(x: torch.Tensor, edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None) -> torch.Tensor
~~~

Tầng nhận trọng số adjacency thô. Tính cạnh chuẩn hóa bằng `compute_symmetric_norm`, chiếu `support = x @ weight`, rồi gọi `sparse_spmm`. Nếu bật bias, cộng **sau phép tổng hợp**: $\hat A(XW)+b$. Phép chiếu tuyến tính có bias trước tổng hợp sẽ cho $\hat A(XW+b)$, nhìn chung khác với công thức trên vì $\hat A$ không phải ma trận stochastic theo hàng. Activation và dropout thuộc mạng bao ngoài.

Kiểm chứng trên đồ thị tính tay được, nút cô lập, xử lý loop trùng, dtype của đặc trưng/trọng số, shape đầu ra, gradient hữu hạn theo $x$ và $W$, cùng hành vi reset tường minh. Không cần phụ thuộc `torch_geometric.nn.GCNConv` trong mã mô hình hoặc kiểm thử.

**Quyết định kiến trúc:** yêu cầu bàn giao hiện tại chọn `nn.Module` kết hợp sparse SpMM. Cách này thực hiện cùng tổng thông điệp cục bộ có trọng số về mặt toán học, nhưng không triển khai các hook `MessagePassing.propagate/message` nêu trong kế hoạch trước đây. Ghi nhận thay đổi hướng triển khai trong hồ sơ thiết kế; không tuyên bố đã cài các hook đó.

### Phiên 2 — Ghép mạng `GCN` hai tầng

Sử dụng `GCN(nn.Module)` với chữ ký tương thích trainer là `forward(x, edge_index)`. Khởi đầu bằng số chiều **1433 → 16 → 7** và:

$$
H=\operatorname{Dropout}\!\left(
\operatorname{ReLU}(\hat AXW^{(0)}+b^{(0)})
\right),
\qquad
Z=\hat AHW^{(1)}+b^{(1)}.
$$

Dùng dropout 0,5 và trả logits thô; cross-entropy tự xử lý log-softmax. Trong phiên bản đầu ưu tiên tính đúng, truyền **cùng cạnh thô** cho hai tầng, mỗi tầng tự tính chuẩn hóa. Không truyền hệ số đã chuẩn hóa của tầng trước vào hàm của tầng sau nếu hàm đó yêu cầu trọng số thô.

Việc tính lại có chi phí chuẩn bị đo được. Tối ưu sau này có thể chuẩn hóa một lần và cung cấp đường xử lý nội bộ mang tên rõ ràng cho input đã chuẩn hóa; giữ đường này tách khỏi hợp đồng input thô. Ban đầu không dùng cache qua các lần chạy. Cache bổ sung về sau phải xét topology, trọng số, số nút, dtype, device và giai đoạn split, không chỉ shape tensor hoặc số cạnh.

Kiểm chứng đầu ra `[2708,7]`, tính xác định ở chế độ evaluation, dropout hoạt động khi training, gradient qua cả hai tầng và khả năng overfit một tập train nhỏ. GCN và GAT vẫn là hai mạng riêng, không ghép hai loại tầng vào một mô hình.

### Phiên 3 — Tích hợp trainer và chốt protocol so sánh

Sau thiết lập seed deterministic, tạo GCN tự cài đặt, Adam với learning rate 0,01, weight decay $5\times10^{-4}$ và cross-entropy. Tải Cora đã chuẩn hóa với public masks không thay đổi. Truyền `edge_index` vô hướng thô cho trainer, dùng checkpoint và log riêng, ví dụ `results/checkpoints/gcn_seed42/` và `results/gcn_seed42_log.json`.

Khởi đầu với tối đa 200 epoch và patience 20. Ghi lịch sử validation, xác minh khôi phục checkpoint cùng cách ly mask trên dữ liệu giả và chỉ tuning bằng validation. Lần chạy tích hợp Tuần 5 không cho phép kiểm tra test lặp lại; giữ mốc so sánh Tuần 7 theo kế hoạch và chỉ đánh giá test khi cấu hình mô hình tương ứng đã chốt. Mọi mở rộng phải được xác định trước.

Mục tiêu tham khảo là khoảng **81,5% test accuracy**, theo kết quả GCN trong [Kipf và Welling, Bảng 3](https://arxiv.org/pdf/1609.02907#page=7). Đây là mốc tài liệu, không phải điểm được bảo đảm, trần hiệu năng hoặc ngưỡng nghiệm thu.

Đối chiếu với [log MLP đã chốt](../../results/baseline_mlp_log.json):

| Metric | MLP đã chốt | GCN tự cài đặt |
|---|---:|---|
| Test Accuracy | 56,80% | Chờ đánh giá cuối; mục tiêu tham khảo khoảng 81,5% |
| Test Macro-F1 | 53,87% | Chờ đánh giá cuối |

Nếu đạt mục tiêu, chênh lệch accuracy tuyệt đối là **24,70 điểm phần trăm**, không phải tăng tương đối 24,70%. Báo cáo mức cải thiện thực đo bằng $100(a_{\mathrm{GCN}}-0.568)$ và $100(f_{\mathrm{GCN}}-0.5386504117428538)$. RQ1 chưa được giải quyết về thực nghiệm trước khi GCN tự cài đặt được đánh giá dưới cùng split và protocol lựa chọn. Diễn giải mức tăng cùng các khác biệt kiến trúc và giới hạn của một seed; quy kết mạnh hơn cho cấu trúc đồ thị cần ablation adjacency có kiểm soát, được lên kế hoạch trước, không phải thử nghiệm dựa trên test.
