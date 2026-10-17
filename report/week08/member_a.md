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


### GCN final đã hoàn tất

Selection được commit trước final tại `0cf2917c`. Lệnh thực tế (Python 3.11.9 runtime local):

```powershell
runs/python311/python.exe -m scripts.nc_week08 final-gcn --run runs/week08-a/search --output-dir runs/week08-a/final-gcn
```

Restore đúng best checkpoint candidate 3, xác minh SHA-256 checkpoint/protocol,
features/edges/train-val masks và source hashes khớp selection. Không retrain NC
trên train+val. Claim `runs/week08-a/search/final_claim.json` được tạo exclusive
trước đọc nhãn test; mọi lần gọi sau trên cùng run đều bị từ chối, kể cả đổi output.
Claim còn nguyên khi lỗi; không tự retry. Đây là guard theo run, không phải khóa toàn
repo: tuyệt đối không tạo run mới để chấm lại final. Bản sao claim được lưu cùng kết quả.

[Final summary](../../results/week08/final-gcn/summary.json),
[predictions/logits](../../results/week08/final-gcn/predictions.npz),
[masks](../../results/week08/final-gcn/masks.npz),
[bảng so sánh](../../results/week08/comparison.csv).
Đã kiểm tra lại metrics bằng predictions đã lưu và hash file, không forward model
hoặc đọc lại dataset để đánh giá lần hai.

| Model | Test Accuracy | Test Macro-F1 | Nguồn |
|---|---:|---:|---|
| MLP | 56,80% | 53,87% | Final lịch sử, không chạy lại |
| GCN | 82,10% | 81,07% | Final một lần, 1.000 test nodes |
| GAT | Chờ B | Chờ B | Chưa có cấu hình đã tuning |

RQ1: GCN hơn baseline MLP 25,30 điểm phần trăm Accuracy và 27,21 điểm Macro-F1,
phù hợp với lợi ích của topology trong thiết lập Cora này. Một seed, budget/capacity
khác MLP và không có ước lượng phương sai giới hạn kết luận tổng quát.
RQ2 chưa có so sánh final GCN/GAT; không dùng validation GAT tuần 7 thay test.
Không thay đổi search hoặc ablation sau khi thấy điểm test.

A14: phần GCN hoàn tất kỹ thuật và sẵn sàng review; toàn task vẫn Doing do thiếu
GAT final. MLP reuse đúng số lịch sử; check-in 2 và danh sách ablation chung cần B
xác nhận/bàn giao. Không ghi nhận nghiệm thu cá nhân hoặc check-in đã diễn ra.
