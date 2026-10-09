# Tự kiểm tra GAT Layer triển khai sớm

**Ngày chạy:** 2026-10-09

**Phạm vi:** Một GAT Layer sparse, chạy trước lịch Tuần 6. Không xây mạng GAT hoàn chỉnh, không huấn luyện/đánh giá trên Cora và không chấm test.
**Review:** Không giao người A review hoặc benchmark. Kết quả dưới đây là kiểm tra tự động do trợ lý chạy theo yêu cầu; chưa ghi nhận xác nhận thủ công của người dùng.

## Cài đặt

Implementation nằm ở [models/gat_layer.py](../../models/gat_layer.py), softmax theo nút nhận ở [utils/attention.py](../../utils/attention.py), và các kiểm tra ở [tests/test_gat_layer.py](../../tests/test_gat_layer.py). API nhận `x` cùng `train_adjacency` theo chiều `source_to_target`; tùy chọn trả `(output, (processed_edges, alpha))`, trong đó `alpha` được lấy trước attention dropout. Self-loop đầu vào được bỏ rồi thêm lại đúng một lần cho mỗi nút.

Layer hỗ trợ nhiều attention head, nối head hoặc lấy trung bình, bias, dropout attention và dropout đặc trưng riêng. Attention và aggregation chỉ chạy trên cạnh sparse; không tạo ma trận attention dày. PyG `GATConv` chỉ được dùng trong test làm mốc đối chiếu với trọng số và quy ước tương đương.

## Kết quả

Môi trường kiểm tra: Python 3.11, PyTorch 2.7.1+cpu, PyTorch Geometric 2.8.0.post1; CUDA không khả dụng.

| Lệnh | Kết quả |
|---|---|
| `.venv\Scripts\python.exe -m pytest tests\test_gat_layer.py -q` | 22 passed |
| `.venv\Scripts\python.exe -m pytest tests -q` | 85 passed, 1 skipped (kiểm tra CUDA) |

Các test GAT bao gồm softmax nhóm ổn định số học và đối chiếu PyG, shape một/nhiều head, tổng attention theo nút nhận, self-loop duy nhất, gradient hữu hạn, hoán vị cạnh, dropout theo train/eval, graph rỗng/nút cô lập, input không hợp lệ, so sánh đầu ra với `GATConv` khi đồng bộ trọng số, và overfit một graph toy.

## Giới hạn

Kết quả này chỉ xác nhận layer trên graph tổng hợp và regression suite hiện có. Chưa đo tốc độ, chưa ghép thành encoder hai tầng, chưa tích hợp node classification hoặc link prediction, và chưa đưa ra kết luận hiệu năng trên Cora. Mục test CUDA được bỏ qua vì môi trường dùng bản PyTorch CPU.
