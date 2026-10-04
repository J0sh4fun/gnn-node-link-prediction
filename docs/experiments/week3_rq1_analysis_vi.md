# Tuần 3, Phiên 3 — Baseline MLP và phân tích RQ1

**Thí nghiệm:** phân loại nút bán giám sát trên Cora; MLP hai tầng chỉ sử dụng đặc trưng; seed 42.  
**Nguồn bằng chứng:** [log kết quả cuối cùng](../../results/baseline_mlp_log.json), [script huấn luyện](../../train/train_mlp.py), [mô hình](../../models/mlp.py) và bản tóm tắt thí nghiệm do Thành viên A cung cấp. Kết quả được báo cáo là có thể tái lập trong môi trường cố định; tài liệu này không thay thế một lần chạy xác minh độc lập.

## 1. Cập nhật bảng tiến độ tại Mục 4

Bảng dưới đây cập nhật kết quả Node Classification tại [Mục 4 của kế hoạch](../../Plan_GNN.md). Các giá trị là **chỉ số trên tập test ở lần đánh giá cuối cùng**, biểu diễn theo phần trăm. Chỉ số validation phục vụ lựa chọn checkpoint và không được điền thay cho kết quả test.

| Mô hình | Test Accuracy (%) | Test Macro-F1 (%) | Trạng thái |
|---|---:|---:|---|
| MLP — baseline chỉ dùng đặc trưng | **56,80** | **53,87** | Đã chốt, Tuần 3, seed 42 |
| GCN — tự cài đặt | — | — | Cài tầng ở Tuần 5; so sánh ở Tuần 7 |
| GAT — tự cài đặt | — | — | Cài tầng ở Tuần 6; so sánh ở Tuần 7 |

Giá trị lưu trong JSON là accuracy `0.568` và macro-F1 `0.5386504117428538`; việc làm tròn trong bảng không thay đổi dữ liệu lưu trữ. Dấu gạch ngang biểu thị thí nghiệm chưa hoàn thành, không phải điểm bằng không hoặc kết quả ước lượng.

Baseline sử dụng 1.433 đặc trưng đầu vào, 16 đơn vị ẩn, bảy lớp đầu ra, ReLU, dropout 0,5 và Adam với learning rate 0,01, weight decay 0,0005. Split Planetoid chuẩn gồm 140 nút train, 500 nút validation và 1.000 nút test. Đặc trưng đầu vào được chuẩn hóa theo hàng; “chỉ dùng đặc trưng” có nghĩa bộ phân loại không nhận ma trận kề, không có nghĩa dữ liệu chưa qua tiền xử lý.

## 2. Phân tích chặt chẽ RQ1

**RQ1: Thông tin cấu trúc đồ thị có cải thiện hiệu năng phân loại nút so với chỉ sử dụng đặc trưng bag-of-words hay không, và cải thiện bao nhiêu?**

**Kết luận hiện tại:** thí nghiệm xác lập mốc đối chiếu chỉ dùng đặc trưng; câu trả lời thực nghiệm của dự án vẫn cần chờ kết quả GCN/GAT tự cài đặt. So sánh với tài liệu hỗ trợ kỳ vọng về lợi ích của cấu trúc đồ thị, nhưng chưa đo được mức cải thiện do đồ thị trong repository này.

### Giới hạn của đặc trưng từ vựng

Biểu diễn Cora gốc là vector nhị phân thưa về sự xuất hiện của từ, sau đó được loader chuẩn hóa L1. Vì vậy, mô tả chính xác là đặc trưng hiện diện từ dạng bag-of-words đã chuẩn hóa, thay vì số đếm tần suất từ thô. Các cơ chế sau là giải thích lý thuyết cho những sai số có thể xảy ra, chưa phải kết luận từ một phân tích lỗi đã hoàn thành:

- **Nhập nhằng từ vựng:** các từ như “network” hoặc “learning” xuất hiện trong nhiều lĩnh vực. Bag-of-words loại bỏ thứ tự từ và ngữ cảnh, khiến MLP phải suy ra khác biệt từ các mẫu đồng xuất hiện thưa.
- **Đồng nghĩa:** các bài báo giải quyết vấn đề tương tự có thể sử dụng thuật ngữ khác nhau. Những tọa độ từ vựng riêng biệt không trực tiếp biểu diễn quan hệ tương đương ngữ nghĩa; việc học quan hệ này từ chỉ 20 nút có nhãn mỗi lớp là khó khăn.
- **Giao thoa liên ngành:** các lớp khác nhau có thể chia sẻ phương pháp và thuật ngữ. Ngược lại, các bài báo cùng lớp có thể tập trung vào những nhánh nghiên cứu khác nhau, tạo ra biến thiên từ vựng đáng kể trong nội bộ lớp.

MLP hai tầng có thể học tổ hợp phi tuyến giữa các từ; do đó, không nên diễn giải các giới hạn trên thành việc mô hình hoàn toàn không học được quan hệ ngữ nghĩa. Giới hạn xác định của baseline là dự đoán cho mỗi nút chỉ phụ thuộc vector đặc trưng của chính nút đó và bộ tham số dùng chung. Mô hình không trực tiếp xem xét các bài báo lân cận theo quan hệ trích dẫn hoặc đặc trưng của chúng khi suy luận.

### Thiên kiến quy nạp từ mạng trích dẫn

Giả thuyết nghiên cứu là các bài báo liên kết bởi quan hệ trích dẫn thường chia sẻ chủ đề, phương pháp hoặc bối cảnh học thuật. Với tính đồng nhất nhãn theo liên kết này, tức homophily, việc tổng hợp thông tin láng giềng có thể bổ sung ngữ cảnh khi từ vựng của một bài báo còn mơ hồ hoặc thưa. GCN sử dụng phép tổng hợp có chuẩn hóa, trong khi GAT học trọng số attention phụ thuộc láng giềng; cả hai đều có thể kết hợp nội dung với thông tin quan hệ.

Tín hiệu quan hệ là **bổ sung**, chưa được chứng minh trực giao hay độc lập thống kê với vector từ. Hai nút có cùng vector từ nhận cùng dự đoán MLP ở chế độ suy luận xác định, nhưng mô hình đồ thị có thể phân biệt chúng nhờ các vùng lân cận khác nhau. Quan hệ trích dẫn cũng có thể nối các bài báo ít trùng từ vựng. Những cơ chế này không đòi hỏi truy cập nhãn bị giữ lại để đánh giá.

Homophily là giả định quy nạp, không phải bảo đảm cho từng cạnh. Trích dẫn khác chủ đề và liên kết nhiễu có thể đưa thông điệp không phù hợp vào biểu diễn. Do đó, lợi ích phải được đo dưới quy ước đồ thị vô hướng thống nhất, thay vì mặc nhiên suy ra từ sự tồn tại của các cạnh.

### Đối chiếu tài liệu và khoảng cải thiện kỳ vọng

[Kipf và Welling (ICLR 2017), Bảng 3](https://arxiv.org/pdf/1609.02907#page=7) báo cáo accuracy trên Cora là **55,1% cho MLP** và **81,5% cho GCN sử dụng phép tái chuẩn hóa**. Đây là trung bình của nhiều lần khởi tạo, trong khi **56,80%** của dự án là kết quả một seed cố định. Chênh lệch số học so với MLP trong paper là **+1,70 điểm phần trăm**, chưa chứng minh ưu thế có ý nghĩa thống kê hoặc việc tái lập hoàn toàn cấu hình gốc.

Lấy 81,5% làm mốc tham khảo:

```text
Chênh lệch accuracy kỳ vọng = 81,50% − 56,80%
                           = 24,70 điểm phần trăm.
```

Như vậy, nếu GCN đạt mốc này, mức cải thiện tuyệt đối dự kiến là khoảng **24–25 điểm phần trăm**. Đây không phải mức tăng tương đối 24–25%; mức tăng tương đối của accuracy tương ứng xấp xỉ 43,5%.

**81,5% là mốc benchmark, không phải trần hiệu năng của GCN/GAT.** [Veličković và cộng sự (ICLR 2018), Bảng 2](https://arxiv.org/pdf/1710.10903#page=8) báo cáo **83,0 ± 0,7%** cho GAT trên Cora. Giá trị trung tâm này cao hơn baseline hiện tại 26,20 điểm phần trăm. Không điền các số liệu từ tài liệu vào hàng kết quả mô hình tự cài đặt và không coi chúng là ngưỡng đạt/không đạt của đồ án.

Ở Tuần 5–7, mức cải thiện đo được cần được báo cáo theo:

```text
ΔAccuracy_pp = 100 × (Accuracy_GNN − 0.568)
ΔMacroF1_pp  = 100 × (MacroF1_GNN − 0.5386504117428538)
```

Giữ nguyên split, chuẩn hóa đặc trưng, hàm tính metric và nguyên tắc chọn mô hình chỉ bằng validation. Ghi rõ khác biệt kiến trúc và regularization khi diễn giải kết quả. Một ablation thay đổi adjacency trong khi giữ các yếu tố khác tương đương sẽ hỗ trợ quy kết lợi ích cho cấu trúc đồ thị tốt hơn việc đồng thời thay đổi nhiều lựa chọn kiến trúc. Nếu bổ sung seed hoặc ablation, phải xác định thiết kế trước các lần đánh giá test cuối cùng, không lựa chọn hồi cứu theo điểm test.

## 3. Động lực huấn luyện và kiểm toán protocol

### Lựa chọn checkpoint và điều kiện dừng

| Đại lượng | Giá trị quan sát | Diễn giải |
|---|---:|---|
| Epoch validation tốt nhất | 183 / 200 | Theo bản tóm tắt thí nghiệm của Thành viên A |
| Validation loss nhỏ nhất | 1,330289 | Giá trị JSON: 1.3302887678146362 |
| Epoch dừng thực tế | 200 | Đạt giới hạn số epoch |
| Patience | 20 | Đã cấu hình nhưng chưa dùng hết sau epoch 183 |
| Validation accuracy tại checkpoint được chọn | 58,80% | Không nhất thiết là accuracy lớn nhất trong mọi epoch |
| Validation macro-F1 tại checkpoint được chọn | 56,82% | Checkpoint được chọn theo loss, không chọn riêng theo F1 |
| Test accuracy cuối cùng | 56,80% | Đánh giá sau khi khôi phục checkpoint được chọn |
| Test macro-F1 cuối cùng | 53,87% | Cùng checkpoint đã khôi phục |

Validation loss đạt giá trị nhỏ nhất khá muộn, tại epoch 183. Kết quả này phù hợp với việc tối ưu vẫn mang lại lợi ích ở giai đoạn sau của quá trình huấn luyện, nhưng riêng vị trí cực tiểu không chứng minh đường validation giảm đều hoặc đơn điệu. JSON hiện không lưu loss từng epoch hoặc trường `best_epoch`; thông tin epoch 183 đến từ bản tóm tắt được cung cấp. Cần lưu console trace hoặc lịch sử từng epoch để thực hiện phân tích đường cong sau này.

**Hiệu chỉnh nguyên nhân dừng:** từ epoch 184 đến 200 chỉ có **17** epoch. Với patience 20, lần chạy kết thúc vì **giới hạn 200 epoch**, không phải vì patience đã cạn. Nếu tiếp tục không cải thiện, điều kiện patience sẽ kích hoạt tại epoch 203, ngoài ngân sách đã cấu hình. Không cần huấn luyện thêm hoặc đánh giá lại test để ghi nhận đúng lần chạy đã hoàn thành.

### Khả năng khái quát và overfitting

Validation accuracy cao hơn test accuracy **2,00 điểm phần trăm**. Chênh lệch macro-F1 xấp xỉ **2,95 điểm phần trăm**. Các khoảng cách vừa phải này phù hợp với hiệu năng tương đối gần nhau trên hai tập giữ lại và tự chúng không cho thấy sự suy giảm nghiêm trọng từ validation sang test.

Tuy nhiên, chúng **không chứng minh mô hình không bị overfitting nghiêm trọng**. Overfitting chủ yếu được đánh giá qua tương quan giữa kết quả train và dữ liệu giữ lại; JSON được cung cấp chưa lưu đường train hoặc toàn bộ đường validation. Validation còn tham gia chọn checkpoint, nên mức lạc quan nhất định so với test chưa sử dụng là có thể xảy ra. Một seed và các tập đánh giá khác nhau trên cùng đồ thị cũng chưa đủ xác lập ý nghĩa thống kê. Kết luận có cơ sở là chưa có bằng chứng rõ ràng về sự suy giảm nghiêm trọng từ các chỉ số giữ lại đã báo cáo; chẩn đoán mạnh hơn cần bổ sung dữ liệu về quá trình huấn luyện.

### Kiểm toán protocol và giới hạn bằng chứng

Mục tiêu **không rò rỉ dữ liệu** được hỗ trợ bởi các thuộc tính sau của mã nguồn đã rà soát:

| Cơ chế kiểm soát | Bằng chứng triển khai | Phạm vi bảo đảm |
|---|---|---|
| Cách ly nhãn huấn luyện | Cross-entropy chỉ sử dụng `train_mask` | Nhãn validation/test không tham gia supervised loss |
| Chọn mô hình chỉ bằng validation | Mỗi epoch dùng `eval()`, `no_grad()`, `val_mask`; chọn loss validation nhỏ nhất | Không dùng metric test cho early stopping |
| Bảo toàn checkpoint | `deepcopy(state_dict())` giữ bản sao độc lập; `load_state_dict()` khôi phục | Cập nhật sau đó không ghi đè trạng thái tốt nhất trong bộ nhớ |
| Cách ly test nghiêm ngặt | `fit_mlp()` không nhận test mask; `main()` đánh giá `test_mask` một lần sau khôi phục | Một lần đánh giá cuối cùng trong mỗi lần gọi script |
| Baseline chỉ dùng đặc trưng | `MLP.forward(x)` không nhận adjacency | Việc loader tạo đồ thị vô hướng không biến MLP thành mô hình đồ thị |
| Tiền xử lý không phụ thuộc nhãn | Chuẩn hóa theo từng nút; giữ nguyên mask của public split | Không ước lượng tham số chuẩn hóa từ nhãn test |
| Ghi JSON nguyên tử | Đóng file tạm trong cùng thư mục đích rồi thay thế file kết quả; từ chối giá trị JSON không hữu hạn | Tránh công bố file JSON mới chỉ được ghi một phần theo cơ chế thay thế thông thường |

[Bộ test protocol huấn luyện](../../tests/test_train_mlp.py) kiểm tra khôi phục checkpoint, xử lý patience, thứ tự đánh giá cuối cùng và ghi log trên dữ liệu giả. Đây là kiểm toán triển khai, không phải chứng minh không có leakage trong mọi thao tác thủ công hoặc mọi lần chạy lịch sử. Ghi log nguyên tử bảo vệ tính toàn vẹn của file; riêng cơ chế này không chứng minh tính hợp lệ thống kê hoặc ngăn leakage.

Kết quả test MLP đã chốt phải được giữ cố định trong các giai đoạn phát triển tiếp theo. Script chỉ giới hạn một lần đánh giá test trong mỗi lần gọi, vì vậy quy trình nhóm cũng phải ngăn việc chạy lại nhiều lần để tinh chỉnh theo test. Cần lưu trữ JSON, checkpoint đã chọn, cấu hình và phiên bản môi trường trước khi một lần chạy mới ghi đè các đường dẫn kết quả hiện tại.

## 4. Bàn giao và đồng bộ với Thành viên B

### Trạng thái hàm đánh giá node classification

Triển khai dùng chung nằm tại [`utils/evaluate.py`](../../utils/evaluate.py), với kiểm thử tại [`tests/test_evaluate.py`](../../tests/test_evaluate.py). Hàm thực hiện argmax, chọn nhãn thật và dự đoán theo cùng mask, trả về số thực Python qua hai khóa `accuracy` và `macro_f1`. Macro-F1 sử dụng `average="macro", zero_division=0`; lớp vắng trong cả nhãn thật lẫn dự đoán đã mask không tham gia trung bình theo mặc định của scikit-learn.

**Trạng thái bàn giao đã xác nhận: tỷ lệ đạt 100% trên các test đã thực thi**, căn cứ console pytest do Thành viên A cung cấp: **7 passed, 1 skipped trong 126,42 giây**. Cả **sáu trường hợp node classification trên CPU đều đạt**. Test đạt thứ bảy xác nhận stub link prediction chưa triển khai; test CUDA được bỏ qua do không có CUDA. Tài liệu ghi nhận bằng chứng thực thi đã cung cấp; chưa hoàn thành một lần chạy xác minh độc lập trong môi trường soạn tài liệu.

Bộ kiểm thử gồm sáu trường hợp node classification trên CPU, một trường hợp CUDA sẽ bỏ qua trên máy chỉ có CPU và một test xác nhận stub link prediction phát sinh `NotImplementedError`. Báo cáo riêng số passed, failed và skipped. “Đạt 100%” có nghĩa mọi test phù hợp đã thực thi đều đạt, không có nghĩa đạt 100% code coverage hoặc test CUDA bị bỏ qua đã chạy thành công.

Lệnh xác minh và lưu báo cáo sau đây không huấn luyện lại hoặc đánh giá lại tập test Cora:

```sh
python -m pytest tests/test_evaluate.py -v -ra --junitxml=results/test_evaluate_junit.xml
```

### Đặc tả bàn giao cho Tuần 4

Thành viên B triển khai `evaluate_link_prediction(pos_pred, neg_pred)` trong module hiện có để kiểm tra hạ tầng chia cạnh và đánh giá trong Tuần 4. Công việc chuẩn bị này không đưa thời điểm đánh giá test cuối cùng của link prediction lên sớm hơn kế hoạch.

1. Nhận các tensor điểm positive và negative một chiều, không rỗng, hữu hạn và trên cùng thang đo; điểm cao hơn phải thể hiện bằng chứng mạnh hơn về sự tồn tại cạnh. Xác định và kiểm thử cách báo lỗi cho input rỗng hoặc chứa giá trị không hữu hạn.
2. Detach tensor, chuyển sang CPU, ghép điểm dự đoán và tạo nhãn nhị phân tương ứng: một cho positive, không cho negative.
3. Tính `sklearn.metrics.roc_auc_score` và `sklearn.metrics.average_precision_score` từ **điểm liên tục**, không argmax hoặc đặt ngưỡng. Average Precision không phải phép xấp xỉ PR-AUC bằng quy tắc hình thang.
4. Trả đúng cấu trúc `{"roc_auc": float, "average_precision": float}`. Giữ nguyên API và hành vi của hàm node classification.
5. Thay test kiểm tra ngoại lệ của stub bằng các test metric có kết quả xác định: xếp hạng hoàn hảo, xếp hạng đảo ngược, điểm bằng nhau, số positive/negative khác nhau và input không hợp lệ. Giữ nguyên mọi regression test của node classification.
6. Đặt trách nhiệm chia dữ liệu ngoài hàm metric: hai chiều của cùng cặp vô hướng phải thuộc một split; positive validation/test không xuất hiện trong adjacency khi tuning. Negative phải không tồn tại trong toàn bộ đồ thị gốc và không trùng giữa các split, với tỷ lệ positive/negative cố định 1:1 theo kế hoạch.
7. Khi tuning, đánh giá cạnh validation bằng embedding chỉ sinh từ train adjacency. Giữ cạnh test cho protocol cuối cùng; retrain trên train + validation là giai đoạn riêng theo Mục 3.3.

Đồng bộ tên metric, chiều ý nghĩa của điểm dự đoán, tỷ lệ negative sampling và mã định danh split trước khi ghép bảng kết quả của hai thành viên. Thay đổi các quy ước này sau khi xem kết quả test sẽ làm suy giảm tính so sánh của thí nghiệm.
