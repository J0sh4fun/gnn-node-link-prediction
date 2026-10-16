# Quy tắc làm việc trong repo

## Cập nhật todo khi hoàn thành mỗi job

Áp dụng cho mọi job code, tài liệu hoặc kiểm tra tiến độ trong repo:

1. Hoàn tất đầu ra và kiểm tra phù hợp; ghi bằng chứng thực tế của task.
2. Cập nhật trạng thái/nội dung task có liên quan trong todo A/B. Task đã xong kỹ thuật nhưng chưa merge vẫn là Review; Done yêu cầu nghiệm thu và tích hợp `main`. Với task không có quy tắc tự kiểm chứng, cập nhật theo bằng chứng đã xác minh trước khi đồng bộ.
3. Trước commit cuối và trước khi kết thúc job, chạy `python -m scripts.sync_todos --github --write` với thông tin đăng nhập GitHub được cung cấp an toàn qua môi trường. Script đồng bộ trạng thái đã có bằng chứng, checkbox và tổng số task. Không ghi token vào file hoặc commit.
4. Đưa thay đổi todo vào cùng commit/nhánh của job; khi job có PR thì cập nhật cùng PR đó. Không tạo PR riêng chỉ để đồng bộ todo. Nếu trạng thái PR thay đổi sau commit, đồng bộ lại trong lần kết thúc job hoặc công việc tiếp theo.
5. Nếu không đọc được dữ liệu tiến độ mới, ghi rõ phần đồng bộ bị chặn; không dùng snapshot cũ để khẳng định tiến độ hiện tại.

Không chạy đồng bộ theo lịch hằng ngày, không tạo workflow/nhánh bot/PR định kỳ cho todo. Không tự phê duyệt hoặc merge PR.

Chi tiết và cách mở rộng bằng chứng: [docs/plans/todo_sync.md](docs/plans/todo_sync.md).

## Phân công tuần 5–6

Tuần 5 do A thực hiện toàn bộ GCNLayer, kiểm thử, nghiệm thu, báo cáo và PR.
Tuần 6 do B thực hiện toàn bộ GATLayer, kiểm thử, benchmark, nghiệm thu, báo cáo và PR.
Hai tuần này không có task hoặc điều kiện phê duyệt chéo giữa A và B.
