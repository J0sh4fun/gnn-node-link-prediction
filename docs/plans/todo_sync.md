# Đồng bộ todo theo tiến độ project

Cơ chế đọc trạng thái PR trên GitHub và bằng chứng nghiệm thu đã lưu để cập
nhật checkbox, trạng thái, ngày thay đổi và tổng số task trong hai tracker A/B.
Không chạy thí nghiệm Cora, không đánh giá lại test set và không tự tạo số liệu.

## Cách chạy và lịch tự động

```powershell
$env:GITHUB_TOKEN = '<token đọc repo, không ghi vào file>'
python -m scripts.sync_todos --github --write
```

Bỏ `--write` để kiểm tra: exit 1 nếu todo cần cập nhật, exit 0 nếu đã khớp.
Không in token vào output hoặc đưa token vào commit. Có thể dùng
`--save-snapshot runs/todo-sync/snapshot.json` để lưu dữ liệu API không chứa
token, rồi chạy lại offline bằng `--snapshot runs/todo-sync/snapshot.json`.

Workflow [sync-todos.yml](../../.github/workflows/sync-todos.yml) chạy khi có
push vào `main`, mỗi ngày **09:15 Asia/Bangkok (02:15 UTC)**, hoặc chạy thủ công.
Lịch GitHub có thể bị trễ. Workflow bắt đầu hoạt động khi được merge vào `main`.
Nó chạy tests đồng bộ, commit/push lên nhánh riêng `automation/todo-sync` và
tạo/cập nhật một PR vào `main`; không tự approve/merge. Nhánh này dành riêng
cho bot, không chỉnh tay vì mỗi lần đồng bộ sẽ thay bằng bản mới dựa trên
`main` với force-with-lease.

Repo cần bật **Allow GitHub Actions to create and approve pull requests** để
token của workflow tạo PR. Cơ chế chỉ tạo/cập nhật PR, không gọi API approve.
Quyền job: `contents: write`, `pull-requests: write`; không đổi quyền mặc định
của workflow khác. Xem [GitHub Docs về quyền Actions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository).

## Điều kiện cập nhật

Quy tắc trong [todo_sync.json](../../configs/todo_sync.json) hiện theo dõi tự
động **A06/A07** qua PR #4, **B06-01/B06-02** qua PR #6. Với mỗi task:

1. Đọc PR và đúng base repo/nhánh `main`; phân biệt merged, open, draft và đóng không merge.
2. Kiểm tra JUnit đúng số testcase, không failure/error/skip; có báo cáo nghiệm thu. GAT còn kiểm tra sanity trace đúng commit và finite outputs/gradients.
3. So sánh blob code/test/dependency liên quan giữa commit đã kiểm chứng và PR head; nếu đã merge thì so với `main` hiện tại. Code/test thay đổi hoặc revert làm task chuyển Doing để yêu cầu xác minh lại.
4. Chỉ khi bằng chứng hợp lệ, code khớp và PR đã merge mới chuyển Done. PR đang mở và đủ bằng chứng là Review; draft/đóng không merge hoặc bằng chứng hết hiệu lực là Doing.

Thiếu bằng chứng không được coi là hoàn tất. Lỗi API hoặc tracker sai cấu trúc
dừng đồng bộ trước khi ghi file. Không có thay đổi thực chất thì không đổi
ngày/commit, tránh PR lặp mỗi ngày.

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
- Workflow pass actionlint 1.7.12 (tải bản chính thức, kiểm tra SHA256); không chạy workflow GitHub trước khi merge vào main.
