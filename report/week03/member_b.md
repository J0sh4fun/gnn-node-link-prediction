# Tuần 3 — Cosine baseline trên validation

**Thành viên B; cập nhật 2026-09-28; branch `feature/gat-link-prediction`; trạng thái: kỹ thuật đã chạy, chờ B tự review.**

## Mục tiêu và phương pháp

Baseline tính `cos(x_u,x_v)=dot(x_u,x_v)/max(||x_u||·||x_v||, epsilon)` trên raw Cora features, chỉ cho cặp validation; vector 0 được score 0 theo quy ước. Hàm tính theo chunk, không tạo ma trận N×N. ROC-AUC và Average Precision dùng score liên tục; AP không đồng nhất diện tích PR theo hình thang. Negative pool được cố định từ tuần 2, cân bằng 1:1.

## Kết quả và kiểm chứng

[W3.1](../../notebooks/03_link_prediction/03_cosine_baseline.ipynb) nạp manifest/audit đã pass, kiểm tra fingerprint và chạy toy tests cho cosine, perfect/reversed/tied ranks, vector 0, NaN/Inf, input rỗng, index ngoài miền và mismatch. Trên 263 positive + 263 negative validation pairs: **ROC-AUC 0,8121701918; AP 0,8248011097**. [Result JSON](../../results/week03/cosine_validation.json) ghi run ID, config, split hash và checks. W3.1 được chạy hai lần trong kernel mới; scores/metrics trùng trong tolerance `atol=1e-12`, `rtol=1e-10` (trạng thái `independent_run_reproduction=passed`). Trường test để rỗng, không sinh test predictions.

Run được chọn `20260928T070135_w3_cosine_a81fd148`, đối chiếu với `20260928T070124_w3_cosine_5c6533e6`; split ID `cora_lp_v1`, hash `9febb7a31cc7f0362d6c4d39bf7c7a37d6af96a3b9f2ea0f8c7b8630e2cb8c98`, seed 42.

[W3.2](../../notebooks/03_link_prediction/04_cosine_analysis.ipynb) đọc run predictions và dựng [biểu đồ score](figures/cosine_score_distribution.png). Phân bố positive/negative có overlap, nên cosine chưa tách hoàn toàn liên kết thật/giả. AP phụ thuộc prevalence; giá trị ở đây thuộc validation cân bằng 1:1, không đại diện mọi cặp node có thể có. Đây là baseline không học; chưa có GAT hoàn chỉnh để so sánh.

Theo [RQ3 của kế hoạch nhóm](../../docs/plans/project_plan.md), kết quả này là mốc để sau này so embedding GCN/GAT với đặc trưng gốc theo cùng split và protocol. Nó chưa trả lời RQ3 khi chưa có model học.

## Tự kiểm tra và bước tiếp

Agent hỗ trợ tạo code, chạy hai run và đối chiếu. B chưa xác nhận đã tự chạy/giải thích. B cần trình bày: vì sao score liên tục cần cho ROC-AUC/AP; vì sao zero vector được quy ước 0; vì sao validation không thay final test. Tuần 4 dùng framework toy để kiểm tra kỹ thuật; final test chỉ làm ở đợt đánh giá đã chốt.

Nguồn: [ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [config](../../configs/cosine_baseline.json).
