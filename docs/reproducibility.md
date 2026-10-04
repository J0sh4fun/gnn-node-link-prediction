# Chạy lại từ checkout mới

1. Cài Python 3.11, tạo `.venv` và cài `requirements.txt` theo README. Nếu máy không có CUDA, cài bản PyTorch CPU tương thích trước khi cài các gói còn lại; `torch==2.14.0` trong môi trường đã kiểm chứng là build `+cu132`.
2. Mở notebook bằng VS Code với kernel của `.venv`, hoặc cài JupyterLab riêng và mở từ root repo. Chạy W2.1 → W2.2 → W2.3 → W3.1 (hai lần vào hai run mới) → W3.2; W4.1 và W4.2 độc lập; cuối cùng W4.3. Danh sách đầy đủ tại [notebooks/README.md](../notebooks/README.md).
3. W2.1 sẽ tải Cora vào `data/cora/` qua PyG Planetoid. `data/cora/` cần quyền ghi và kết nối tải dataset ở lần đầu. Fingerprint sau tải phải khớp manifest của split đã commit.
4. Mỗi notebook chạy bằng **Restart Kernel and Run All**; không dựa biến RAM từ notebook trước. W3.2 cần run local của W3.1 để đọc dự đoán validation. Clone mới chạy W3.1 trước.
5. Kết quả nhỏ được chọn trong `results/`; run đầy đủ trong `runs/<run_id>/` bị Git ignore. `results/experiment_index.csv` ghi provenance. Test metric trống cho tới final evaluation.

Các notebook nạp helper `.ipynb` bằng IPython `%run`. Nguồn: [IPython `%run`](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-run). Khi thiếu artifact, notebook báo file và bước cần chạy, không tự tạo split thay thế.
