# Cora local

W2.1 dùng `torch_geometric.datasets.Planetoid(root="data/cora", name="Cora", split="public")` và tự tải dữ liệu nếu chưa có cache. Thư mục `data/cora/` được Git ignore. Không đổi thứ tự node hoặc thêm nhãn vào feature. Mỗi lần chạy so fingerprint feature/full canonical edges với `artifacts/splits/cora_lp_v1/manifest.json` trước khi dùng split.
