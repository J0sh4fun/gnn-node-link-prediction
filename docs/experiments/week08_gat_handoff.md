# A13 — Nhận cấu hình GAT đã tuning

Ngày 2026-10-17, API GitHub xác nhận PR #10/#11 đã merge (84f2b56a/5e1ff82c).
GCN search 11 cấu hình và GAT search 8 cấu hình đều chọn minimum best_val_loss.
[Hợp đồng GAT](../../results/week08/gat_handoff.json) cố định config 7:
hidden 16/head, 8 hidden heads, 1 output head, 2 tầng, lr 0,005,
wd 5e-4, feature dropout **0,4** và attention dropout **0,4**, seed 42;
max 200 epoch, patience 20, Cora public split và normalized features.
GAT depth cố định 2; search của B không khảo sát depth, không tự mở rộng sau test.

Kiểm tra lịch sử `c430f494:models/gat.py` và `models/gat_layer.py` cho thấy
`dropout` của B điều khiển attention, `feature_dropout` điều khiển features.
Merge 4524cb23 đổi sang layer MessagePassing đã nghiệm thu nhưng default
attention_dropout 0,6 không còn giữ đúng ý nghĩa API B khi dropout=0,4.
Sửa constructor để lấy giá trị attention trước khi resolve feature alias;
caller truyền `attention_dropout` tường minh vẫn được ưu tiên.
Không thay layer GATLayer đã nghiệm thu tuần 6.

Reconstruction ban đầu với attention=0,6 bị dừng trước test: val loss 0,71043044,
Accuracy 0,808, Macro-F1 0,79637626. Giữ audit lần thất bại; không dùng test
để phát hiện hoặc chọn cách sửa. Sau sửa theo source B, vẫn cùng config 7,
validation khớp **chính xác**: epoch 189, loss 0,6794008016586304,
Accuracy 0,812, Macro-F1 0,8012218832929454.

Tests kiểm tra hai quy ước dropout, explicit override và backward/shape/trainer.
Suite cuối: **145 passed, 1 skipped CUDA**; xem
[JUnit](../../results/week08/completion_junit.xml).

Giữ nguyên 5 ablation trong protocol đã định trước: GCN self-loop/depth/hidden (A),
GAT heads/LP decoder (B). Chưa chạy tuần 10; không đổi phạm vi theo điểm final.
A13 hoàn tất kỹ thuật, Review chờ tích hợp bản sửa dropout/handoff vào main.
