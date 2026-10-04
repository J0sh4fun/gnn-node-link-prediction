# Quyết định triển khai phần B

| Ngày | Quyết định | Lý do |
|---|---|---|
| 2026-09-28 | Giữ implementation, helper và test trong `.ipynb` | Phù hợp định dạng thực hành của kế hoạch; branch B độc lập. |
| 2026-09-28 | Split canonical pair, negatives tự lấy từ pool sparse bằng NumPy RNG | Bảo đảm đủ số lượng 1:1 sau canonicalization, seed CPU cố định, không nhầm `negative_sampling(force_undirected=True)` trả cột hai hướng. |
| 2026-09-28 | Dùng `data/cora/` làm root Planetoid | Đường dẫn không phụ thuộc máy; trên Windows tên cũ `data/Cora` tương đương về chữ hoa/thường. |
| 2026-09-28 | Cosine raw float64, epsilon `1e-12`, chunk 256 | Không học tham số, ổn định với vector 0, tránh N × N similarity. |
| 2026-09-28 | Tuần 4 chỉ triển khai ProjectedSum và framework toy | GAT đầy đủ/attention training theo lịch tuần 6. |
