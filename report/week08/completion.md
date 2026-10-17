# Mốc tuần 8 — Đối chiếu GCN/GAT và hoàn tất đầu ra kỹ thuật

Ngày kiểm chứng: **2026-10-17**. A13/A14: **Review**, chờ review/merge bản hoàn tất.
Đã đủ cấu hình, kết quả NC và artifact; **có ngoại lệ protocol GAT chạy lại test theo yêu cầu người dùng**.
Không khẳng định đáp ứng nguyên tắc test một lần cho GAT; không ghi nhận cuộc họp
check-in 2 hoặc phê duyệt cá nhân đã diễn ra. Báo cáo A/B trước đây giữ làm snapshot lịch sử.

## Đầu vào đã xác minh

API GitHub mới xác nhận [PR #10](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/10)
merge `84f2b56a` và [PR #11](https://github.com/J0sh4fun/gnn-node-link-prediction/pull/11)
merge `5e1ff82c`. [Snapshot trạng thái](../../results/week08/pr_status.json).
GCN search 11 cấu hình giữ nguyên; GAT search 8 cấu hình của B nhận qua
[gat_tuning.json](../../results/week08/gat_tuning.json).
Hai bên dùng seed 42, Cora public split, row-normalized features,
train 140/validation 500/test 1.000 nodes, graph vô hướng đầy đủ theo NC transductive.
Checkpoint và config chọn bằng minimum val_loss; không retrain NC trên train+val.

| Mô hình | Cấu hình được chốt | Best epoch | Val loss | Val Accuracy | Val Macro-F1 |
|---|---|---:|---:|---:|---:|
| GCN | depth 3, hidden 16, dropout 0,5, lr 0,01, wd 5e-4 | 103 | 0,70508945 | 77,40% | 76,02% |
| GAT | depth 2, hidden 16/head, 8 heads, 1 output head, feature/attention dropout 0,4, lr 0,005, wd 5e-4 | 189 | 0,67940080 | 81,20% | 80,12% |

[Handoff A13](../../docs/experiments/week08_gat_handoff.md) ghi chi tiết lỗi mapping
dropout sau merge và cách sửa dựa trên source B. Layer MessagePassing đã nghiệm thu
được giữ nguyên. Tái lập validation sau sửa khớp chính xác toàn bộ chỉ số của config 7.
Giữ danh sách 5 ablation/degree bins đã định trước trong protocol GCN;
không chạy ablation tuần 10 hoặc thay phạm vi theo điểm test.

## Ngoại lệ GAT và khôi phục artifact

B đã công bố GAT final Accuracy 82,20%/Macro-F1 81,31668880445113%.
Checkpoint được nêu trong báo cáo B không có ở workspace; lần final cũ chưa lưu
predictions/logits. Người dùng trả lời rõ **“chạy lại test”**, cho phép ngoại lệ
đối với protocol một lần trong plan. Không ghi lại lịch sử thành chỉ có một lần final.

Chỉ dựng lại config 7 đã chọn, không search, không đổi seed/hyperparameter theo test.
Trong fit, nhãn ngoài train/validation bị thay -999. Lần dựng lại đầu dùng attention
0,6 bị dừng vì validation sai, **chưa đánh giá test**; audit giữ tại
[failed_reconstruction](../../results/week08/final-gat/failed_reconstruction/failure_audit.json).
Sau khi sửa đúng attention 0,4 theo source B, validation khớp chính xác rồi mới restore
checkpoint và thực hiện **một lần đánh giá lặp được cho phép**. Kết quả khớp final B.
Dấu claim lần thất bại được lưu trước khi mở attempt đã sửa; claim attempt thành công
vẫn còn trên đĩa. Guard cho phép audit thủ công lỗi, không tự retry hoặc ghi đè.

[Final GAT summary](../../results/week08/final-gat/summary.json),
[config](../../results/week08/final-gat/config.json),
[versions/source hashes](../../results/week08/final-gat/environment.json),
[predictions/logits](../../results/week08/final-gat/predictions.npz),
[masks](../../results/week08/final-gat/masks.npz),
[repeat claim](../../results/week08/final-gat/repeat_claim.json).
Checkpoint local (ignored): `runs/week08-gat-recovery-corrected/best_gat_val.pt`;
SHA-256 `922fdc58b7f906d6d3f37914188bce0a06d3e72dabc154503ac8440d843b4449`.
Weights không nằm trong Git theo quy ước repo; không chạy lại final để tái kiểm tra.
Run lưu revision nền trước commit và dirty=true; source hashes xác định code thực thi.

## Bảng final và RQ1/RQ2

| Model | Test Accuracy | Test Macro-F1 | Ghi chú |
|---|---:|---:|---|
| MLP | 56,80% | 53,87% | Dùng lại final lịch sử |
| GCN | 82,10% | 81,07% | Giữ nguyên final một lần ở PR #10 |
| GAT | 82,20% | 81,32% | Đánh giá lặp được người dùng cho phép, khớp final B |

[Bảng CSV](../../results/week08/comparison.csv),
[audit từ predictions đã lưu](../../results/week08/completion_audit.json),
[experiment index](../../results/experiment_index.csv).
Đã đối chiếu hash predictions/masks, metrics từ predictions, argmax logits, node IDs,
1.000 samples, cùng toàn bộ masks và hash features/edges/train-val giữa GCN/GAT.
Bước audit không forward model và không đánh giá lại trên dataset.

RQ1: GCN/GAT hơn MLP 25,30/25,40 điểm phần trăm Accuracy và 27,21/27,45 điểm Macro-F1.
RQ2: GAT hơn GCN 0,10 điểm Accuracy (một node), khoảng 0,2455 điểm Macro-F1.
Chỉ một seed, khác depth/hidden capacity/search budget và không có ước lượng phương sai;
không đủ kết luận attention tốt hơn tổng quát hoặc khác biệt có ý nghĩa thống kê.
GAT final đã được lặp theo ngoại lệ; báo cáo giữ cả kết quả gốc và kết quả khôi phục,
không dùng chúng để thay đổi cấu hình/ablation.

## Kiểm tra và trạng thái mốc

**145 passed, 1 skipped CUDA**, không failure/error:
[JUnit](../../results/week08/completion_junit.xml), [log](../../results/week08/completion_pytest.txt).
Tests mới bảo vệ semantics dropout B và explicit override của A, chặn grid search khi
đã có final dù đổi output, chặn retry attempt lỗi, poison nhãn ngoài train/val,
chặn test khi validation không khớp, xuất predictions và claim bền qua output khác.

Lệnh kiểm thử (không chạy Cora final):

```powershell
python -c "import torch,pytest; torch.set_num_threads(1); torch.manual_seed(42); raise SystemExit(pytest.main(['tests','-q']))"
```

Lệnh khôi phục đã thực hiện với sự cho phép của người dùng, **không chạy lại để smoke**:

```powershell
runs/python311/python.exe -m scripts.recover_gat_final --acknowledge-repeat --output-dir runs/week08-gat-recovery-corrected
```

A13/A14 đủ đầu ra kỹ thuật, Review chờ tích hợp các sửa lỗi và artifact mới.
Điều kiện protocol một lần của GAT có ngoại lệ rõ ràng ở trên; không tự xác nhận
mốc đạt protocol nguyên bản, không tự approve/merge PR. Tuần 9–12 vẫn To do.
