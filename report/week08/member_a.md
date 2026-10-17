# Tuần 8 — Tuning và final NC của A

Ngày kiểm chứng: **2026-10-17**. Nhánh `feature/a-week08-nc-tuning`.

## Tiến độ và phụ thuộc

API GitHub xác nhận PR #9 đã merge vào main (`7074f7d3`, 2026-10-16T11:56:08Z).
A10–A12 chuyển Done; tuần 1–7 của A đã tích hợp. B06-01–B06-04 đã có đầu ra tích hợp.
A13/A14 giữ Doing vì chưa nhận cấu hình GAT đã tuning của B; không lấy cấu hình tuần 7 thay cho bàn giao tuning.
Người dùng đồng ý tiếp tục phần GCN và ghi rõ phụ thuộc GAT. Không tự approve/merge.

## A13 — Search GCN và đóng phạm vi ablation

[Protocol định trước](../../configs/node_classification_week08.json) được ghi trước khi fit:
11 cấu hình one-factor, seed 42, CPU một thread, tối đa 200 epoch/run, patience 20.
Thay đổi lần lượt lr 0,005/0,01/0,02; depth 2/3/4; hidden 16/32/64;
dropout 0,3/0,5/0,7; weight decay 0/0,0005/0,001. Không phải tích Descartes;
không mở rộng search sau khi xem validation. Cùng preprocessing/split public Cora như tuần 7.
Checkpoint và cấu hình đều chọn **minimum validation loss**, tie theo thứ tự candidate.
Không dùng Accuracy hoặc test để đổi lựa chọn. Search hoàn toàn không đọc nhãn held-out;
`test_score` null trong mọi run.

[11 kết quả](../../results/week08/search.csv), [selection đóng băng](../../results/week08/search/selection.json).
Candidate 3 thắng: 3 tầng, hidden 16, dropout 0,5, lr 0,01, wd 0,0005,
best epoch 103, val loss 0,70508945; Accuracy 77,40%, Macro-F1 76,02%.
Giá trị Accuracy không cao nhất không thay đổi tiêu chí lựa chọn đã định trước.
GCN mở rộng depth >=2, giữ conv1/conv2 và API hai tầng tương thích tuần 7.

Chốt 5 ablation tuần 10 trong protocol: GCN self-loop, depth, hidden (A);
GAT heads và LP decoder (B, danh sách đề xuất chờ phối hợp B). Depth đo thêm
mean pairwise cosine hidden embeddings trên cùng nodes để khảo sát over-smoothing.
Error analysis dùng validation: degree không self-loop, bins 0–2/3–5/6+,
số mẫu/số lỗi/tỷ lệ lỗi và confusion matrix. Không chạy các ablation trong job tuần 8.

## Kiểm chứng và tái lập

**136 passed, 1 skipped (CUDA)**, không failure/error: [JUnit](../../results/week08/junit.xml),
[log](../../results/week08/pytest.txt). Tests tổng hợp kiểm tra depth 2/3/4 có finite gradients,
đổi nhãn held-out thành -999 vẫn giữ nguyên search/selection, checkpoint/graph bị đổi
bị từ chối trước claim và final không chạy lần hai dù đổi output directory.

Môi trường .venv mất Python gốc; đã tải Python 3.11.9 embeddable chính thức vào
`runs/python311/` và dùng thư viện sẵn có ở `.venv/Lib/site-packages`:
PyTorch 2.7.1+cpu/PyG 2.7.0. Không thay dependency hoặc commit runtime.
Versions, revision nền, dirty flag và source SHA-256 nằm trong từng environment.json;
source hashes xác định code thực thi trên working tree trước commit.

Với Python 3.11 và requirements đã cài, từ root repo:

```powershell
python -m scripts.nc_week08 tune --output-dir runs/week08-reproduce/search
```

Checkpoint local: `runs/week08-a/search/candidate_03/best_model.pt` (ignored).
Config/environment/history của mọi candidate nằm trong `results/week08/search/`;
checkpoint SHA-256 và graph hash trong selection. Chạy lại tuning cần thư mục mới;
không chạy final để smoke hoặc tái kiểm tra tài liệu.

## A14 — Final

Cấu hình GCN đã đóng băng; final evaluator sẵn sàng. GAT chờ B bàn giao
search space/budget/seed, metric selection, config, checkpoint/hash, versions,
preprocessing/split và validation history. Sau đó A kiểm tra tương thích rồi mới
chấm GAT một lần. Check-in 2 chưa hoàn tất khi thiếu GAT final.
