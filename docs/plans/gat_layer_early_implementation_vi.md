# Kế hoạch triển khai sớm GAT Layer

Tài liệu này là kế hoạch làm trước phần GAT Layer của Tuần 6. Nó bổ sung hướng dẫn thực hiện cho cá nhân, không thay đổi lịch, phân công hoặc deliverable trong [kế hoạch dự án gốc](project_plan.md). Phạm vi chỉ gồm một layer GAT tự cài đặt và tự kiểm tra; chưa gồm mạng GAT hoàn chỉnh, huấn luyện node classification/link prediction, tuning hay benchmark tốc độ.

**Trạng thái:** Đã triển khai sớm ngày 2026-10-09. Bằng chứng chạy và giới hạn được ghi trong [báo cáo tự kiểm tra](../../report/week06/gat_layer_self_check.md).

## Mục tiêu và đầu ra

Tự cài đặt layer theo công thức GAT ở Mục 2.3 của kế hoạch gốc, dùng PyTorch và các thao tác tensor/sparse phù hợp của repository. Không gọi `GATConv` trong đường tính chính. Cuối đợt cần có implementation, kiểm tra tái chạy được và ghi chép trung thực các tiêu chí đạt/chưa đạt.

Đầu ra dự kiến theo kế hoạch gốc:

- `models/gat_layer.py`: `GATLayer`.
- `tests/test_gat_layer.py`: kiểm tra tự động trên đồ thị nhỏ và input tổng hợp.
- Một ghi chú kết quả ngắn, ví dụ `report/week06/gat_layer_self_check.md`, chỉ tạo sau khi thực sự chạy kiểm tra.

Trước khi tạo module, kiểm tra các lớp hiện có trong `layers/` và quy ước import/package để đặt implementation nhất quán. Nếu chọn đường dẫn khác với `models/gat_layer.py`, ghi lại quyết định và giữ nguyên API xuyên suốt kiểm thử.

## Hợp đồng đề xuất

Theo thiết kế đã chuẩn bị ở notebook Tuần 4, bắt đầu với giao diện:

```python
GATLayer(
    in_channels,
    out_channels,
    heads=1,
    concat=True,
    dropout=0.0,  # attention dropout
    feature_dropout=0.0,
    negative_slope=0.2,
    add_self_loops=True,
    bias=True,
)

forward(x, train_adjacency, return_attention=False)
```

- `x`: `[num_nodes, in_channels]`.
- `train_adjacency`: edge list dạng `[2, num_edges]`, quy ước `source_to_target`; ở link prediction chỉ truyền adjacency train. Attention softmax được nhóm theo nút nhận.
- `dropout` áp dụng lên attention sau softmax; `feature_dropout` áp dụng riêng lên đặc trưng đầu vào.
- Khi `concat=True`, đầu ra là `[num_nodes, heads * out_channels]`; khi `False`, head được lấy trung bình và đầu ra là `[num_nodes, out_channels]`.
- Mặc định chỉ trả tensor đầu ra. Khi yêu cầu attention, trả thêm cạnh sau xử lý self-loop và hệ số attention trước attention-dropout để tổng theo mỗi nút nhận có thể được kiểm tra.
- Không thêm activation bên trong layer trừ khi công thức/hợp đồng hiện tại yêu cầu; mạng ngoài chịu trách nhiệm activation giữa các layer.

Trước khi code, đối chiếu tên đối số, quy ước cạnh và output với notebook [GAT design/check-in](../../notebooks/05_model_design/01_gat_design_and_checkin.ipynb) và công thức Mục 2.3. Hợp đồng notebook quy định khi yêu cầu attention trả `(embedding, (processed_edges, alpha))`. Nếu phát hiện điểm không khớp khác, ghi quyết định ở đầu implementation/test để tránh âm thầm thay đổi hợp đồng.

## Các chặng triển khai

### 1. Chốt phép tính bằng ví dụ nhỏ

- Viết ra chiều tensor sau phép chiếu tuyến tính cho từng head.
- Với từng cạnh `j → i`, tính score từ biểu diễn nguồn và đích, áp dụng LeakyReLU, rồi chuẩn hóa softmax trên toàn bộ cạnh đi vào cùng `i`.
- Tính thông điệp `alpha_ij * projected_x_j`, tổng hợp theo nút nhận, sau đó nối hoặc trung bình các head.
- Chọn một graph toy có ít nút, một nút có nhiều láng giềng, một nút cô lập và một phép tính tay đủ nhỏ để kiểm tra chiều cạnh, self-loop và nhóm softmax.

### 2. Cài forward đơn head

- Thêm tham số học được, khởi tạo Xavier/Glorot và bias bằng 0 nếu bật.
- Thêm self-loop theo cấu hình, loại self-loop cũ trước khi thêm để không nhân đôi.
- Tính attention score trên cạnh sparse; không tạo ma trận attention dày `[N, N]`.
- Tính grouped softmax theo nút nhận sao cho ổn định số học; giữ hệ số trước dropout để trả về và kiểm tra.
- Tổng hợp thông điệp và xác nhận shape đầu ra cho một head.

### 3. Mở rộng multi-head và dropout

- Hỗ trợ `heads > 1`.
- Hỗ trợ nối head và lấy trung bình head với shape theo hợp đồng.
- Tách dropout đặc trưng khỏi attention dropout. Attention trả về phục vụ kiểm tra là hệ số trước dropout; message passing dùng hệ số sau dropout khi `training=True`.
- Xác nhận `eval()` cho kết quả lặp lại, còn dropout chỉ tác động khi `training=True`.

### 4. Viết và tự chạy kiểm tra

Kiểm tra trên đồ thị toy, không tải Cora ở các phép kiểm tra đơn vị:

- Shape đầu ra với một/nhiều head, concat bật/tắt, có/không bias.
- Với mỗi nút nhận có cạnh, tổng attention trước dropout xấp xỉ 1; nút chỉ có self-loop cũng có attention bằng 1.
- Self-loop được thêm đúng một lần; graph có cạnh lặp hoặc nút cô lập không làm phát sinh NaN/Inf.
- Forward và backward chạy được; gradient hữu hạn tồn tại cho input và mọi tham số cần học.
- Permutation các thứ tự cạnh không làm đổi đầu ra trong tolerance số học.
- `eval()` xác định; các kiểm tra dropout phân biệt rõ train/eval.
- Input sai shape, index ngoài miền hoặc dtype không hỗ trợ được từ chối bằng lỗi dễ hiểu.

### 5. Kiểm tra tham chiếu và khả năng học

- Đối chiếu với `torch_geometric.nn.GATConv` trên graph rất nhỏ, chỉ như oracle kiểm tra: đồng bộ trọng số, bias, self-loop, hướng cạnh, head, concat, activation và dropout. Ghi rõ khác biệt nếu cấu hình không thể tương đương; không ép equality khi implementation convention khác.
- Tạo một bài overfit nhỏ, cố định seed, loss và optimizer; xác nhận loss giảm rõ ràng trên cùng một tập toy. Đây là sanity check, không phải đánh giá benchmark.
- Tự chạy toàn bộ kiểm tra liên quan đến GAT và ghi lại command, seed, phiên bản phụ thuộc, kết quả, sai khác, mục nào chưa đạt. Không dùng kết quả validation/test của dự án để chọn layer hoặc tuning trong kế hoạch này.

## Tiêu chí hoàn thành

- [x] Implementation chính không gọi `GATConv`.
- [x] Hợp đồng đầu vào/đầu ra và quy ước attention được ghi rõ.
- [x] Forward/backward, shape, self-loop, multi-head và grouped softmax đạt kiểm tra trên graph toy.
- [x] Output và gradient hữu hạn; gradient xuất hiện cho input và các tham số học được.
- [x] Kiểm tra đối chiếu `GATConv` có cấu hình tương đương đã chạy.
- [x] Sanity check overfit toy đạt.
- [x] Báo cáo tự kiểm tra ghi lệnh chạy và giới hạn; các kiểm tra chưa thực hiện được nêu rõ.

## Ngoài phạm vi

Không triển khai GAT hai tầng, không chạy benchmark tốc độ, không huấn luyện trên Cora, không chấm test, không tích hợp link prediction và không sửa lịch Tuần 6 trong kế hoạch dự án gốc. Việc review của người khác không phải điều kiện để hoàn tất kế hoạch cá nhân này; bạn tự chạy và tự ghi kết quả.
