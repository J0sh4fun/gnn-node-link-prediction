# Protocol dữ liệu và đánh giá — B

- Cora `Planetoid(split="public")` giữ nguyên thứ tự node, `data.x` là feature duy nhất cho cosine. Public train/val/test masks dành cho node classification; LP tạo edge split riêng.
- Đồ thị LP vô hướng, sparse edge list `[2,M]`. Canonical pair có `u<v`; bỏ self-loop và duplicate trước split. `edge_index[0]` là source, `edge_index[1]` là target khi message passing.
- `configs/data_protocol.json` là nguồn seed/tỷ lệ: seed 42, validation `floor(0.05M)`, test `floor(0.10M)`, train lấy phần còn lại. Một negative unique cho mỗi positive, không thuộc **full** positive graph, không giao giữa các split.
- Encoder adjacency chỉ từ `train_pos` chuyển hai chiều. Self-loop thêm đúng một lần tại layer, không lưu trong split. Full graph được đọc khi tạo negative exclusion/audit, không làm adjacency tuning.
- `edges.npz` chứa sáu mảng int64 shape `[2,M]`: `train_pos`, `val_pos`, `test_pos`, `train_neg`, `val_neg`, `test_neg`. `manifest.json` có hash từng mảng, hash tổng theo tên/dtype/shape/bytes, fingerprint feature và full canonical positives. Hash không phụ thuộc metadata của file NPZ.
- Rerun W2.2 chỉ đối chiếu split hiện có; đổi seed/protocol phải tạo split ID mới. Audit W2.3 nạp lại file từ đĩa và làm kiểm tra đối chứng lỗi.
- Cosine chỉ chấm `val_pos` và `val_neg`. ROC-AUC/AP dùng scores liên tục và negatives cố định. Test pairs chỉ được đọc để audit giao tập; không sinh test scores/metrics trong giai đoạn này. Kết quả toy tuần 4 gắn `dataset=toy`.

Nguồn API: [Planetoid](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html), [MessagePassing](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html), [ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
