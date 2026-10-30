import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Credit Scoring & XAI Assistant",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Trợ Lý Duyệt Tín Dụng Thông Minh (XAI)")
st.caption("Real-time Credit Risk Assessment & Explainable AI (TreeSHAP)")

# Đường dẫn API có thể cấu hình linh hoạt qua biến môi trường (hỗ trợ Docker Compose)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# Dữ liệu dự phòng mặc định (Fallback) khi clone repo chưa có file parquet
FALLBACK_SAMPLE = {
    "pctinstlsallpaidlat10d_839L": 0.1047,
    "numinstlsallpaid_934L": 100.0,
    "credamount_770A": 40000.0,
    "age": 67.0,
    "requesttype_4525192L": "UNKNOWN",
    "maxdpdlast24m_143P": 0.0,
    "avgdbddpdlast24m_3658932P": -11.0,
    "days180_256L": 0.0,
    "mobilephncnt_593L": 3.0,
    "numberofqueries_373L": 0.0,
    "totaldebt_9A": 98567.22,
    "incometype_1044T": "RETIRED_PENSIONER",
    "education_927M": "P97_36_170",
    "sex_738L": "F",
    "maxdbddpdtollast12m_3658940P": -2.0,
    "maxdpdfrom6mto36m_3546853P": 0.0,
    "amtinstpaidbefduel24m_4187115A": 175891.2,
    "maxdbddpdlast1m_3658939P": -9.0,
    "maxdpdlast3m_392P": 0.0,
    "applications30d_658L": 0.0,
    "maxdpdlast6m_474P": 0.0,
    "numinstpaidlate1d_3546852L": 14.0,
    "description_5085714M": "2fc785b2",
    "maxdbddpdtollast6m_4187119P": -5.0,
    "days30_165L": 0.0,
    "maxdpdlast12m_727P": 0.0,
    "language1_981M": "P209_127_106",
    "empl_employedtotal_800L": "UNKNOWN"
}

# Đọc thử file parquet nếu có, ngược lại dùng fallback
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parquet_path = os.path.normpath(os.path.join(BASE_DIR, "data", "gold", "test_gold.parquet"))

sample_data = dict(FALLBACK_SAMPLE)
if os.path.exists(parquet_path):
    try:
        df_test = pd.read_parquet(parquet_path)
        sample_data = df_test.drop(columns=['target']).iloc[5].to_dict()
    except Exception:
        sample_data = dict(FALLBACK_SAMPLE)

st.write("Nhập hoặc điều chỉnh thông tin khách hàng để đánh giá rủi ro thời gian thực:")

# Giao diện nhập liệu tương tác
st.subheader("1. Thông tin tài chính khách hàng")
col1, col2 = st.columns(2)

with col1:
    total_debt = st.number_input(
        "Tổng nợ hiện tại (totaldebt_9A)",
        value=float(sample_data.get("totaldebt_9A", 98567.22))
    )
    credit_req = st.number_input(
        "Khoản vay yêu cầu (credamount_770A)",
        value=float(sample_data.get("credamount_770A", 40000.0))
    )

with col2:
    late_ratio = st.slider(
        "Tỷ lệ trả chậm quá 10 ngày (pctinstlsallpaidlat10d_839L)",
        min_value=0.0,
        max_value=1.0,
        value=float(sample_data.get("pctinstlsallpaidlat10d_839L", 0.1047)),
        step=0.01
    )
    age = st.number_input(
        "Độ tuổi khách hàng (age)",
        min_value=18,
        max_value=100,
        value=int(sample_data.get("age", 67))
    )

# Nút bấm đánh giá
if st.button("🚀 ĐÁNH GIÁ HỒ SƠ", type="primary", use_container_width=True):
    # Cập nhật các giá trị người dùng vừa thay đổi vào payload
    payload_data = dict(sample_data)
    payload_data["totaldebt_9A"] = total_debt
    payload_data["credamount_770A"] = credit_req
    payload_data["pctinstlsallpaidlat10d_839L"] = late_ratio
    payload_data["age"] = float(age)

    payload = {"features": payload_data}

    with st.spinner("Đang kết nối tới Inference Engine..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            result = response.json()

            if response.status_code != 200:
                st.error(f"❌ LỖI TỪ SERVER: {result.get('detail', result)}")
            else:
                prob = result["prediction"]["default_probability"]
                decision = result["prediction"]["decision"]
                metadata = result.get("model_metadata", {})

                st.write("---")
                st.subheader("2. Quyết định phê duyệt & Giải thích rủi ro (XAI)")

                st.caption(
                    f"Mô hình: `{metadata.get('active_model')}` (v{metadata.get('version')}) | "
                    f"Ngày huấn luyện: `{metadata.get('train_date')}`"
                )

                if decision == "REJECT":
                    st.error(f"❌ **TỪ CHỐI CHO VAY (REJECT)** — Xác suất vỡ nợ: **{prob * 100:.2f}%**")
                else:
                    st.success(f"✅ **PHÊ DUYỆT / XEM XÉT THÊM (APPROVE)** — Xác suất vỡ nợ: **{prob * 100:.2f}%**")

                col_flags1, col_flags2 = st.columns(2)

                with col_flags1:
                    st.markdown("### 🚨 Yếu tố tăng rủi ro (Red Flags)")
                    red_flags = result["explanations"].get("top_red_flags", [])
                    if red_flags:
                        for flag in red_flags:
                            st.warning(
                                f"**{flag['feature']}** = `{flag['value']}`\n\n"
                                f"Tác động SHAP: **+{flag['impact']:.4f}**"
                            )
                    else:
                        st.info("Không có Red Flag đáng kể.")

                with col_flags2:
                    st.markdown("### 💚 Điểm cộng hồ sơ (Green Flags)")
                    green_flags = result["explanations"].get("top_green_flags", [])
                    if green_flags:
                        for flag in green_flags:
                            st.success(
                                f"**{flag['feature']}** = `{flag['value']}`\n\n"
                                f"Tác động SHAP: **{flag['impact']:.4f}**"
                            )
                    else:
                        st.info("Không có Green Flag đáng kể.")

        except requests.exceptions.ConnectionError:
            st.error(
                f"⚠️ Không thể kết nối tới API tại `{API_URL}`. "
                "Hãy đảm bảo FastAPI server hoặc Docker container đang chạy!"
            )
        except Exception as e:
            st.error(f"Lỗi không xác định: {e}")
