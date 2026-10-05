# Quyết định triển khai phần B

| Ngày | Quyết định | Lý do |
|---|---|---|
| 2026-09-28 | Giữ implementation, helper và test trong `.ipynb` | Phù hợp định dạng thực hành của kế hoạch; branch B độc lập. |
| 2026-09-28 | Split canonical pair, negatives tự lấy từ pool sparse bằng NumPy RNG | Bảo đảm đủ số lượng 1:1 sau canonicalization, seed CPU cố định, không nhầm `negative_sampling(force_undirected=True)` trả cột hai hướng. |
| 2026-09-28 | Dùng `data/cora/` làm root Planetoid | Đường dẫn không phụ thuộc máy; trên Windows tên cũ `data/Cora` tương đương về chữ hoa/thường. |
| 2026-09-28 | Cosine raw float64, epsilon `1e-12`, chunk 256 | Không học tham số, ổn định với vector 0, tránh N × N similarity. |
| 2026-09-28 | Tuần 4 chỉ triển khai ProjectedSum và framework toy | GAT đầy đủ/attention training theo lịch tuần 6. |
| 2026-10-04 | Sau tích hợp Phase 1, đưa helper vào package Python; notebook import implementation chuẩn | Yêu cầu cleanup hiện tại thay thế quy định notebook-only của kế hoạch cá nhân B; giữ adapter và liên kết lịch sử. |
| 2026-10-04 | Giữ nguyên hai root cache, hỗ trợ CLI chỉ rõ cache có sẵn | Đính chính ghi chú cũ: Planetoid tự thêm `Cora/`; root `data/` và `data/cora/` khác một cấp thư mục, không chỉ khác chữ hoa/thường. |
| 2026-10-04 | Không hợp nhất các utility có ngữ nghĩa khác nhau | Epoch, min_delta, dtype/device, cạnh trùng và persistence phải giữ nguyên; xem `docs/phase1_cleanup.md`. |
