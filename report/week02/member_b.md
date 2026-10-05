# Tuần 2 — Cora, split link prediction và audit leakage

**Thành viên B; cập nhật 2026-09-28; branch `feature/gat-link-prediction`; trạng thái: phần kỹ thuật đã chạy, chờ B tự đọc/chạy/giải thích lại.** Revision khi chạy được ghi trong JSON; worktree có thay đổi chưa commit.

## Mục tiêu và kiến thức

Chuẩn hóa Cora thành các cặp vô hướng `u<v`, chia positive 85/5/10 với `floor` cho validation/test, lấy một negative hợp lệ cho mỗi positive và xác nhận encoder adjacency chỉ gồm train positives hai chiều. Public node masks không được dùng làm LP edge split. Một cạnh `(u,v)` và `(v,u)` là cùng một cặp; chia chúng riêng có thể đưa cạnh validation vào train graph.

## Cách thực hiện và kết quả

[W2.1 EDA](../../notebooks/01_data_exploration/02_cora_link_prediction_profile.ipynb) tải Cora public vào `data/cora/`, kiểm tra shape/index, ghi [EDA JSON](../../results/week02/eda_summary.json). Cora có **2.708 node, 1.433 feature, 7 lớp, 10.556 cột cạnh có hướng và 5.278 cặp vô hướng duy nhất**. Không có self-loop, duplicate cột có hướng hay node cô lập trong bản dữ liệu này. Degree trung bình 3,898. Xem [degree](figures/degree_distribution.png) và [class distribution](figures/class_distribution.png).

[W2.2 split](../../notebooks/03_link_prediction/01_edge_split.ipynb) dùng seed 42 và [config](../../configs/data_protocol.json). [Manifest](../../artifacts/splits/cora_lp_v1/manifest.json) và [edges.npz](../../artifacts/splits/cora_lp_v1/edges.npz) lưu split cố định: train 4.488, validation 263, test 527 positive pairs; mỗi phần có số negative bằng positive. Split hash `9febb7a31cc7f0362d6c4d39bf7c7a37d6af96a3b9f2ea0f8c7b8630e2cb8c98`.

[W2.3 audit](../../notebooks/03_link_prediction/02_leakage_audit.ipynb) đọc lại NPZ/manifest từ đĩa, kiểm tra hash/fingerprint, disjoint, full positive union, negative exclusion, train adjacency và tái lập seed. [Audit JSON](../../results/week02/leakage_audit.json) báo `passed`. Đối chứng cố ý thêm validation edge đảo chiều vào adjacency và thay train negative bằng held-out positive đều được phát hiện. Run ID của từng bước nằm trong result JSON/manifest.

Run được chọn: EDA `20260928T070055_w2_eda_9ff70574`, split `20260928T065735_w2_split_66363bd1`, audit `20260928T070114_w2_audit_2c83ad20`. Cấu hình hiệu lực và môi trường của mỗi run nằm trong `runs/<run_id>/` local; bản gọn lưu trong result/manifest.

## Phân tích, giới hạn và tự kiểm tra

Graph full chỉ phục vụ negative exclusion và audit; dùng nó làm adjacency cho encoder sẽ lộ held-out edges. Split test mới được kiểm tra cấu trúc, chưa chấm prediction. Phần agent hỗ trợ tạo notebook, chạy assertion và ghi report. B chưa xác nhận đã tự chạy hoặc giải thích lại. B cần tự trả lời: vì sao 10.556 cột ứng với 5.278 cặp; vì sao negative phải loại toàn bộ positive; vì sao cần test đối chứng leakage. Tuần 3 dùng split/hash này cho cosine validation.

Nguồn: [PyG Planetoid](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html), [protocol](../../docs/protocol.md).
