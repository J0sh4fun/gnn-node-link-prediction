# Tuần 7 — Node Classification của A

Ngày kiểm chứng: **2026-10-16**. Phạm vi: **A10–A12** theo
[kế hoạch A](../../docs/plans/member_a_plan.md). Hoàn tất kỹ thuật, **Review**,
chờ nghiệm thu/tích hợp PR vào `main`; không ghi nhận phê duyệt cá nhân thay A.

## Kiểm tra tiến độ và đầu vào

Đã fetch GitHub trước khi tạo nhánh `feature/a-week07-node-classification`
từ `origin/main` tại `410cf8ec`. Tuần 1–5 có đầu ra tích hợp trên main;
GCN qua PR #4, GATLayer qua PR #6, benchmark/bàn giao tuần 6 qua PR #8
đều đã merge khi kiểm tra. Lệnh đồng bộ GitHub đọc bằng chứng hiện hành
ghi nhận A06/A07 Done và B06-01–B06-04 Done. Các báo cáo audit cũ là lịch sử.
Tuần 7 không còn blocker đầu vào; tuần 8–12 chưa được thực hiện trong job này.

## Đầu ra và nghiệm thu kỹ thuật

| Task | Đầu ra | Bằng chứng |
|---|---|---|
| A10 | [GCN hai tầng](../../models/gcn.py): GCN → ReLU → dropout → GCN | Logits `[N, C]`, finite gradients mọi tham số; mỗi tầng nhận cùng cạnh thô, tự chuẩn hóa |
| A11 | [GAT hai tầng](../../models/gat.py): hidden heads concat, ELU, output heads average | Hidden `[N, hidden_dim * heads]`, logits `[N, C]`; feature/attention dropout theo train/eval; không ghép GCN |
| A12 | [Runner validation](../../scripts/node_classification.py), [config định trước](../../configs/node_classification_week07.json), hai run Cora và bảng MLP/GCN/GAT | Dùng `NodeClassificationTrainer` hiện có; config/versions/hashes/history/seed/best checkpoint được lưu; không gọi final evaluator |

[Tests mới](../../tests/test_nc_networks.py) kiểm tra cách nối hai tầng với phép tính
tham chiếu, shape hidden/output, backward hữu hạn, không sửa cạnh đầu vào,
khôi phục checkpoint và chạy lặp với nhãn ngoài train/validation bị thay thành
`-999`: history/metrics không đổi. Dữ liệu test tổng hợp không có `test_mask`.

Toàn suite: **131 passed, 1 skipped**, không failure/error; skip duy nhất là
`test_cuda_inputs` vì không có CUDA. [JUnit](../../results/week07/junit.xml),
[log](../../results/week07/pytest.txt). Phạm vi nghiệm thu CPU float32;
không tuyên bố đã kiểm chứng GPU.

## Protocol và kết quả Cora

Cùng loader Cora public Planetoid, L1 row-normalized features, topology đầy đủ
vô hướng, 2.708 node/1.433 features/7 classes; loss chỉ trên 140 node train,
validation trên 500 node. Self-loop do từng layer xử lý. Không chấm 1.000 node test.
Hai mô hình được khởi tạo/huấn luyện độc lập, seed 42, CPU một thread,
Adam, tối đa 200 epoch, patience 20, checkpoint chọn bằng **val_loss nhỏ nhất**.
Cấu hình ghi trước khi chạy, không search hoặc thay config sau khi xem kết quả.

| Model | Cấu hình chính | Best epoch | Val Accuracy | Val Macro-F1 |
|---|---|---:|---:|---:|
| MLP lịch sử | hidden 16, dropout 0,5, lr 0,01, wd 5e-4 | Không có trong log cũ | 58,80% | 56,82% |
| GCN | hidden 16, dropout 0,5, lr 0,01, wd 5e-4 | 199 | 77,60% | 76,21% |
| GAT | hidden 8/head, 8 hidden heads, 1 output head; feature/attention dropout 0,6; lr 0,005, wd 5e-4 | 199 | 80,00% | 79,08% |

GCN/GAT đều chạy đủ 200 epoch; best validation loss lần lượt
**0,8147416115 / 0,8234642744** (giá trị chính xác xem summary).
Các file [GCN](../../results/week07/gcn/summary.json),
[GAT](../../results/week07/gat/summary.json), [CSV so sánh](../../results/week07/comparison.csv)
và [experiment index](../../results/experiment_index.csv) là nguồn số liệu.
MLP lấy đúng **validation** từ [log lịch sử](../../results/baseline_mlp_log.json);
không chạy lại MLP hoặc đưa điểm test lịch sử vào bảng validation.

RQ1: GCN và GAT cao hơn MLP lần lượt **18,8 / 21,2 điểm phần trăm Accuracy**,
phù hợp với lợi ích của topology trong thí nghiệm sơ bộ này.
RQ2: GAT cao hơn GCN **2,4 điểm phần trăm Accuracy**, Macro-F1 tăng khoảng
**2,88 điểm phần trăm**. Chưa đủ để kết luận ưu thế tổng quát: chỉ một seed,
hidden capacity/dropout/lr khác nhau, chưa tuning hoặc ước lượng phương sai.
Đây là kết quả validation phục vụ tuần 8, không phải bảng final test.

## Tái lập và checkpoint

Môi trường: Python 3.11.9, PyTorch 2.7.1+cpu, PyG 2.7.0. Versions và SHA-256
sources nằm trong `results/week07/{gcn,gat}/environment.json`. Run được thực hiện
trên working tree có thay đổi; revision nền và `dirty_worktree=true` được giữ
nguyên, source hashes xác định code thực thi. Summary lưu hash features/edges,
hash train/val masks và SHA-256 checkpoint. Hai mô hình có cùng input/split hash.

Từ root repo, với môi trường đã cài requirements và cache Cora đầy đủ:

```powershell
python -m scripts.node_classification --output-dir runs/week07-reproduce
python -c "import torch,pytest; torch.set_num_threads(1); torch.manual_seed(42); raise SystemExit(pytest.main(['tests','-q','--junitxml=results/week07/junit.xml']))"
```

Trên máy này dùng `.venv/Scripts/python.exe` thay `python`. Runner mặc định chạy
cả hai model; dùng `--model gcn` hoặc `--model gat` khi cần riêng. Mỗi lần chạy
phải dùng thư mục mới; runner từ chối ghi đè run cũ và không tự tải dataset.

Checkpoint đã lưu tại `runs/week07-a/gcn/best_model.pt` và
`runs/week07-a/gat/best_model.pt`; cả hai đã đối chiếu bằng tensor với best state
được trainer khôi phục. Thư mục `runs/` và weights bị ignore theo quy ước repo:
weights hiện có ở workspace này, **không có trong Git/PR**. Config, history,
metrics, hash weights và environment được đưa vào Git; dùng lệnh trên để tái tạo
checkpoint trên checkout khác. Không đánh giá final test khi kiểm tra tái lập.

## Bước tiếp theo

A10–A12 giữ **Review** đến khi được nghiệm thu và merge; không tự approve/merge.
Tuần 8: ghi search space/budget/seeds, tuning GCN và nhận config GAT từ B;
chốt ablation trước final. Chỉ khi đã chốt config mới thực hiện final test
theo A14 và lưu predictions. Không dùng bảng tuần 7 làm bằng chứng hoàn tất tuần 8.
