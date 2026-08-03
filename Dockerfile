# 1. Dùng base image Python 3.11 bản slim (nhẹ, tối ưu cho môi trường chạy)
FROM python:3.11-slim

# 2. Cập nhật OS và cài đặt thư viện lõi C++ (Rất quan trọng để XGBoost chạy mượt trên Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# 3. Thiết lập thư mục làm việc mặc định trong Container
WORKDIR /app

# 4. Copy file cấu hình thư viện vào trước để tận dụng Docker Cache
COPY requirements.txt .


# 5. Cài đặt các thư viện Python (Không lưu cache tải về để giảm dung lượng)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy mã nguồn API và các file cấu hình, mô hình vào Container
COPY api/ ./api/
COPY model_artifacts/ ./model_artifacts/

# 7. Thông báo Container sẽ sử dụng cổng 8000
EXPOSE 8000

# 8. Lệnh mặc định sẽ chạy khi Container khởi động
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]