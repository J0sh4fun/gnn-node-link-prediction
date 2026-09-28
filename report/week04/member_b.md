# Tuần 4 — graph utilities, MessagePassing và training framework

**Thành viên B; cập nhật 2026-09-28; branch `feature/gat-link-prediction`; trạng thái: prototype và toy checks đã chạy, chờ B tự review.**

## Mục tiêu và kiến thức

Chuẩn hóa source/target, cặp vô hướng, adjacency hai chiều, self-loop và neighborhood theo node nhận. Prototype `ProjectedSum` chiếu feature rồi cộng message theo target; chưa có attention. Training framework dùng callbacks, train/eval, validation monitor, early stopping và best checkpoint. GAT tuần 6 cần feature projection `[N,K,F′]`, attention logits `[E,K]`, softmax theo node nhận, weighted sum `[N,K,F′]`, rồi concat/mean. [W4.3](../../notebooks/member_b/week04/03_gat_design_and_checkin.ipynb) chốt shape/bias/dropout/attention interface.

## Kết quả và kiểm chứng

[W4.1](../../notebooks/member_b/week04/01_graph_utilities_and_message_passing.ipynb) kiểm tra duplicate, idempotence, self-loop đúng một lần, node cô lập cuối, shape/dtype/index; output prototype được đối chiếu vòng lặp tự tính trên toy graph, edge order permutation và gradient. Xem [graph checks](../../results/week04/graph_checks.json).

[W4.2](../../notebooks/member_b/week04/02_training_framework_validation.ipynb) dùng toy regression `y=2x+1`, kiểm tra optimizer/finite loss/gradient, validation không tính gradient, mode min/max, patience/min_delta. Best epoch **31**, restored validation MSE **0,0003531498**, dừng sau 44 epoch. Một validation sequence điều khiển xác nhận checkpoint được khôi phục từ epoch 1 thay vì epoch cuối 3. Xem [training checks](../../results/week04/training_checks.json); history/checkpoint nằm trong run local.

W4.3 tự tính grouped softmax nhiều head với score lớn, đối chiếu PyG, kiểm tra nhóm một cạnh, gradient và shape concat/mean. [Check-in checks](../../results/week04/checks.json) tổng hợp 16 checks và xác nhận cùng split hash cho EDA/audit/baseline. Validation cosine giữ nguyên ROC-AUC 0,8121701918 và AP 0,8248011097; test rỗng. File check-in liên kết run ID của từng bước. Đây là toy/prototype; chưa có kết luận về hiệu năng GAT.

Run được chọn: graph `20260928T070150_w4_graph_8e183432`, training `20260928T070158_w4_training_dc7d5249`, check-in `20260928T070206_w4_checkin_265c4ce5`. Check-in đối chiếu Cora split hash `9febb7a31cc7f0362d6c4d39bf7c7a37d6af96a3b9f2ea0f8c7b8630e2cb8c98`.

## Vấn đề, giới hạn và bàn giao

Script `scripts/edge_split.py` cũ xử lý negative sampling theo cột nên không được dùng làm protocol mới; split được viết lại trong notebook W2.2 với audit từ đĩa. Không phát hiện lỗi trong các check tuần 4 sau sửa. Run history/checkpoint local bị Git ignore; clone mới chạy notebook để tái tạo. Agent hỗ trợ code/kiểm chứng; B chưa đánh dấu đã tự học/review. B cần giải thích source/target, softmax theo node nhận, min_delta và best checkpoint. Tuần 5–6 có thể dựa vào graph utilities/trainer interface để tự viết attention/GAT hoàn chỉnh, rồi làm final evaluation theo protocol.

Nguồn: [PyG MessagePassing](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html), [PyTorch checkpoint](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [design notebook](../../notebooks/member_b/week04/03_gat_design_and_checkin.ipynb).
