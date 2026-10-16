# Đồng bộ todo khi hoàn thành mỗi job

Cơ chế đọc trạng thái PR trên GitHub và bằng chứng nghiệm thu đã lưu để cập
nhật checkbox, trạng thái, ngày thay đổi và tổng số task trong hai tracker A/B.
Không chạy thí nghiệm Cora, không đánh giá lại test set và không tự tạo số liệu.

## Rule hiện hành — 2026-10-16

Sau mỗi job code, tài liệu hoặc kiểm tra tiến độ, cập nhật todo ngay và đưa
thay đổi vào cùng commit/nhánh của job. Nếu job có PR, dùng cùng PR đó.
**Không chạy theo lịch hằng ngày và không tạo PR riêng để cập nhật todo.**

Rule được lưu trong [AGENTS.md](../../AGENTS.md) để áp dụng cho các lần làm
việc tiếp theo. Khi hoàn tất một job:

1. Chạy kiểm tra phù hợp và lưu bằng chứng; cập nhật nội dung/task thủ công nếu chưa có quy tắc tự kiểm chứng.
2. Chạy script bên dưới với dữ liệu GitHub mới để đồng bộ các task đã khai báo, checkbox và tổng số task.
3. Commit todo cùng phần việc của job, push và cập nhật PR của chính job nếu có. Nếu không có thay đổi tiến độ thì không tạo commit todo thừa.
4. Chỉ ghi Done khi đủ bằng chứng và đã tích hợp main; xong kỹ thuật nhưng chưa merge vẫn là Review.

## Lệnh cuối job

```powershell
$env:GITHUB_TOKEN = '<token đọc repo, không ghi vào file>'
python -m scripts.sync_todos --github --write
```

Bỏ `--write` để kiểm tra: exit 1 nếu todo cần cập nhật, exit 0 nếu đã khớp.
Không in token vào output hoặc đưa token vào commit. Có thể dùng
`--save-snapshot runs/todo-sync/snapshot.json` để lưu dữ liệu API không chứa
token, rồi chạy lại offline bằng `--snapshot runs/todo-sync/snapshot.json`.

Script chỉ ghi các tracker trong working tree; không commit, push, tạo PR,
approve hoặc merge. Agent thực hiện lệnh này theo rule cuối job trong
AGENTS.md, không có tiến trình nền hoặc GitHub workflow chạy theo lịch.
Workflow đồng bộ/lịch 09:15 và bước tạo PR bot trước đây đã được gỡ.

Quyền đọc repo đủ cho script. Tùy chọn Actions tạo/phê duyệt PR đã được bật
trong lần thiết lập trước không còn là điều kiện của rule này; thay đổi lần
này không sửa các cài đặt quyền repo đã được người dùng phê duyệt.

## Điều kiện cập nhật

Quy tắc trong [todo_sync.json](../../configs/todo_sync.json) hiện theo dõi tự
động **A06/A07** qua PR #4, **B06-01/B06-02** qua PR #6. Với mỗi task:

1. Đọc PR và đúng base repo/nhánh `main`; phân biệt merged, open, draft và đóng không merge.
2. Kiểm tra JUnit đúng số testcase, không failure/error/skip; có báo cáo nghiệm thu. GAT còn kiểm tra sanity trace đúng commit và finite outputs/gradients.
3. So sánh blob code/test/dependency liên quan giữa commit đã kiểm chứng và PR head; nếu đã merge thì so với `main` hiện tại. Code/test thay đổi hoặc revert làm task chuyển Doing để yêu cầu xác minh lại.
4. Chỉ khi bằng chứng hợp lệ, code khớp và PR đã merge mới chuyển Done. PR đang mở và đủ bằng chứng là Review; draft/đóng không merge hoặc bằng chứng hết hiệu lực là Doing.

Thiếu bằng chứng không được coi là hoàn tất. Lỗi API hoặc tracker sai cấu trúc
dừng đồng bộ trước khi ghi file. Không có thay đổi thực chất thì không đổi
ngày/commit, tránh tạo thay đổi thừa khi kết thúc job nhiều lần.

Nếu merge diễn ra sau commit cuối, trạng thái Done được ghi lại ở lần đồng
bộ cuối job hoặc lần làm việc kế tiếp; không mở PR todo nền để theo dõi riêng.

## Phần cần xác minh thủ công

Task chưa được khai báo trong `verified_tasks` giữ trạng thái thủ công, nhưng
vẫn được đếm và đồng bộ checkbox theo trạng thái. Đặc biệt **B06-03 benchmark**
và **B06-04 bàn giao** không tự Done chỉ vì merge PR #6 hoặc xuất hiện file
tên benchmark. B cần thực hiện/kiểm chứng rồi cập nhật tracker. Phân công tuần
5 hoàn toàn A và tuần 6 hoàn toàn B vẫn giữ nguyên.

Để thêm task vào đồng bộ: lưu test/báo cáo thực tế, commit đã kiểm chứng, PR,
danh sách đầy đủ source/test/dependency và số test vào config; review bản khai
bằng chứng cùng PR. Config là bản ghi nghiệm thu đã review, không phải công
cụ tự chứng minh báo cáo tùy ý là đúng. Khi code đổi, chạy kiểm tra phù hợp rồi
cập nhật revision và bằng chứng, không chỉ sửa trạng thái.

Báo cáo tuần và nhật ký cũ là snapshot lịch sử; không viết lại chúng. Trạng
thái hiện tại nằm trong checklist và block `todo-sync:summary`.

## Kiểm chứng lần thiết lập — 2026-10-16

- 16 unit tests pass: merge/draft/đóng không merge, code đổi/revert, bằng chứng lỗi, giữ task thủ công, idempotence và không ghi một phần khi tracker lỗi.
- Đồng bộ với API GitHub thật: A06/A07 Done sau PR #4; B06-01/B06-02 Review theo PR #6; A có 7/22 Done, B tuần 6 có 0/4 Done.
- Chạy lại cùng snapshot offline: không có thay đổi, exit 0.
- Script được chạy lại khi hoàn tất job; snapshot offline dùng cho kiểm thử, không thay dữ liệu GitHub mới khi cập nhật tiến độ thực tế.

PR #7 được cập nhật theo rule cuối job, bỏ workflow theo lịch và tạo PR bot.
Không cần chờ lịch hoặc bật thêm quyền Actions để chạy lệnh đồng bộ.
