import streamlit as st
import requests
import json
import pandas as pd
import os

st.set_page_config(page_title="Hệ thống Duyệt Tín Dụng AI", layout="centered")
st.title("🏦 Trợ Lý Duyệt Tín Dụng Thông Minh (XAI)")
st.write("Nhập thông tin khách hàng để đánh giá rủi ro thời gian thực:")

# Đọc thử 1 dòng dữ liệu làm mẫu để điền sẵn vào giao diện cho nhanh
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parquet_path = os.path.normpath(os.path.join(BASE_DIR, "data", "gold", "test_gold.parquet"))

df_test = pd.read_parquet(parquet_path)
sample_data = df_test.drop(columns=['target']).iloc[5].to_dict()

# Tạo giao diện nhập liệu
st.subheader("1. Thông tin tài chính khách hàng")
col1, col2 = st.columns(2)
with col1:
    total_debt = st.number_input("Tổng nợ hiện tại (totaldebt_9A)", value=float(sample_data["totaldebt_9A"]))
    credit_req = st.number_input("Khoản vay yêu cầu (credamount_770A)", value=float(sample_data["credamount_770A"]))
with col2:
    late_ratio = st.slider("Tỷ lệ trả chậm quá 10 ngày (pctinstlsallpaidlat10d_839L)", 0.0, 1.0, float(sample_data["pctinstlsallpaidlat10d_839L"]))
    age = st.number_input("Độ tuổi khách hàng (age)", min_value=18, max_value=100, value=int(sample_data["age"]))

# Nút bấm Đánh giá
if st.button("ĐÁNH GIÁ HỒ SƠ", type="primary"):
    # Cập nhật các giá trị người dùng vừa thay đổi vào dict dữ liệu mẫu
    sample_data["totaldebt_9A"] = total_debt
    sample_data["credamount_770A"] = credit_req
    sample_data["pctinstlsallpaidlat10d_839L"] = late_ratio
    sample_data["age"] = age
    
    # Gửi yêu cầu tới Docker API
    api_url = "http://localhost:8000/predict"
    payload = {"features": sample_data}
    
    with st.spinner("Đang kết nối tới Docker Server..."):
        try:
            response = requests.post(api_url, json=payload)
            result = response.json()
            
            if response.status_code != 200:
                st.error("❌ LỖI TỪ DOCKER SERVER: {}".format(result.get("detail", result)))
            else:
    # Nếu thành công thì mới chạy tiếp
                prob = result["prediction"]["default_probability"]
                decision = result["prediction"]["decision"]
                        
            st.write("---")
            st.subheader("2. Quyết định phê duyệt")
            if decision == "REJECT":
                st.error("❌ TỪ CHỐI CHO VAY (Xác suất vỡ nợ: {}%)".format(round(prob*100, 2)))
                
                # In Red Flags bằng format an toàn
                st.write("**🚨 Lý do từ chối chính (Red Flags):**")
                for flag in result["explanations"]["top_red_flags"]:
                    msg = "- Biến **{}** có giá trị ({}) -> Tăng rủi ro: +{:.4f}"
                    st.write(msg.format(flag['feature'], flag['value'], flag['impact']))
            else:
                st.success("✅ PHÊ DUYỆT / XEM XÉT THÊM (Xác suất vỡ nợ: {}%)".format(round(prob*100, 2)))
                
                # In Green Flags bằng format an toàn
                st.write("**💚 Điểm cộng của hồ sơ (Green Flags):**")
                for flag in result["explanations"]["top_green_flags"]:
                    msg = "- Biến **{}** có giá trị ({}) -> Giảm rủi ro: {:.4f}"
                    st.write(msg.format(flag['feature'], flag['value'], flag['impact']))
                    
        except Exception as e:
            st.error("Không thể kết nối tới Docker Server. Hãy đảm bảo Container đang chạy! Lỗi: {}".format(e))