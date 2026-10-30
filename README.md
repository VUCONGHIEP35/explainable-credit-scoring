# explainable-credit-scoring
1. Project Overview & Business Impact (Tổng quan & Tác động doanh nghiệp)
Vấn đề thực tế: Chấm điểm tín dụng cho nhóm khách hàng chưa có lịch sử tài chính rõ ràng (Unbanked) là bài toán trị giá triệu đô của các ngân hàng. Mô hình giúp giảm tỷ lệ nợ xấu (Nợ nhóm 3, 4, 5 theo chuẩn CIC) mà vẫn tối ưu hóa được tỷ lệ phê duyệt hồ sơ vay.
Mục tiêu hệ thống: Xây dựng một dịch vụ chấm điểm rủi ro thời gian thực (Real-time Inference) có khả năng tự giải thích lý do từ chối/phê duyệt (Explainable AI) bằng SHAP để đảm bảo tuân thủ các quy định khắt khe của ngành tài chính.
2. System Architecture (Kiến trúc hệ thống)
Vẽ hoặc mô tả luồng đi của dữ liệu bằng chữ:
code
Text
Dữ liệu thô (Parquet) ---> Pipeline lọc biến (IV & Correlation) 
                                     | (Trích xuất 28 biến vàng)
                                     v
Streamlit Web UI <--- [FastAPI Server (Docker)] <--- XGBoost Champion (Gini 60.9%)
 (What-if / XAI)     (Đọc config.json tự động)   <--- TreeSHAP (Giải thích lý do)
3. Data Science & Feature Selection Pipeline (Lọc đặc trưng 2 bước)
Bước 1 (Filter): Loại bỏ biến rác. Tính toán chỉ số Information Value (IV) cho toàn bộ biến. Lọc bỏ các biến có 
I
V
<
0.02
IV<0.02
. Xử lý đa tương quan bằng cách lập ma trận tương quan và dùng IV làm trọng tài (giữ lại biến có IV cao hơn).
Bước 2 (Embedded): Đưa 28 biến sạch vào mô hình LightGBM nông (max_depth=3) để xếp hạng độ quan trọng theo Gain và đảm bảo toàn bộ biến giữ lại đều đóng góp tích cực vào mô hình.
4. Model Benchmarking (Đánh giá đối chứng thực tế)
Trình bày bảng so sánh điểm số trên Tập Test gốc (chưa từng được nhìn thấy):
Random Guess Baseline: AUC = 0.5000
Single Feature Baseline (pctinstlsallpaidlat10d): AUC = 0.5749
Logistic Regression Baseline (Lasso L1): AUC = 0.7679 | Gini = 0.5358
XGBoost Core Model (Champion): AUC = 0.8046 | Gini = 0.6093 | KS = 0.4644
Kết luận: XGBoost cải thiện hiệu năng dự báo 4.78% so với mô hình tuyến tính truyền thống nhờ học được các tương tác phi tuyến (non-linear) phức tạp của dữ liệu.
5. Explainable AI (XAI) with SHAP (Mở hộp đen thuật toán)
Chèn biểu đồ Global SHAP Summary Plot để giải thích xu hướng chung của toàn tập dữ liệu (Người trả nợ nhiều kỳ trong quá khứ thì rủi ro thấp, người nợ nhiều và trễ hạn nhiều thì rủi ro cao).
Trình bày cấu trúc JSON giải thích cục bộ (Local Explainability) mà API trả về cho từng khách hàng (Red Flags & Green Flags), chứng minh tính minh bạch của hệ thống.
6. MLOps & Production Architecture (Hạ tầng vận hành)
FastAPI & Uvicorn: Thiết kế API bất đồng bộ hiệu năng cao với cấu trúc lifespan mới nhất.
Champion/Challenger Framework (config.json): Hệ thống đọc cấu hình động. Người quản trị có thể thay đổi mô hình active hoặc ngưỡng rủi ro mà không cần thay đổi hay khởi động lại code API.
Docker Containerization: Đóng gói toàn bộ ứng dụng bằng Docker giúp hệ thống hoạt động cách ly và có thể triển khai ngay lập tức chỉ với 1 dòng lệnh:
code
Bash
docker run -d --name credit-api -p 8000:8000 credit-scoring-api:v1.0