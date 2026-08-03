import pandas as pd
import xgboost as xgb
import shap
import json
import uvicorn
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ==================================================
# 1. KHAI BÁO BIẾN TOÀN CỤC & TÍNH TOÁN ĐƯỜNG DẪN
# ==================================================
MODEL = None
EXPLAINER = None
CONFIG = None
REQUIRED_FEATURES = None
ACTIVE_MODEL_INFO = None

# Lấy đường dẫn gốc của dự án (Thư mục XAI_risk_score, vì file này nằm trong thư mục api)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==================================================
# 2. LIFESPAN: QUẢN LÝ KHỞI ĐỘNG VÀ TẮT SERVER
# ==================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, EXPLAINER, CONFIG, REQUIRED_FEATURES, ACTIVE_MODEL_INFO
    print("🚀 Đang khởi động hệ thống và tải Model Artifacts...")

    try:
        # Sử dụng đường dẫn tuyệt đối để chống lỗi C++ trên Windows
        config_path = os.path.join(BASE_DIR, "model_artifacts", "config.json")
        features_path = os.path.join(BASE_DIR, "model_artifacts", "selected_features.json")

        # 1. Đọc Metadata Configuration
        with open(config_path, "r", encoding="utf-8") as f:
            CONFIG = json.load(f)

        # 2. Đọc danh sách 28 Features Vàng
        with open(features_path, "r", encoding="utf-8") as f:
            REQUIRED_FEATURES = json.load(f)

        # 3. Đọc đúng Mô hình đang Active từ Config
        active_model_key = CONFIG["active_model"]
        ACTIVE_MODEL_INFO = CONFIG["models"][active_model_key]

        # CHUẨN HÓA ĐƯỜNG DẪN MODEL CHO LÕI C++ CỦA XGBOOST
        raw_model_path = ACTIVE_MODEL_INFO["file_path"]
        model_abs_path = os.path.normpath(os.path.join(BASE_DIR, raw_model_path))

        # 4. Tải Mô hình và Khởi tạo SHAP
        if "xgboost" in active_model_key.lower():
            MODEL = xgb.XGBClassifier(enable_categorical=True)
            MODEL.load_model(model_abs_path)  # Lúc này XGBoost sẽ nhận D:\XAI_risk_score\...
            EXPLAINER = shap.TreeExplainer(MODEL)
        else:
            import joblib
            MODEL = joblib.load(model_abs_path)
            EXPLAINER = None

        print(f"✅ HỆ THỐNG ĐÃ SẴN SÀNG!")
        print(f"-> Đang chạy Model  : {active_model_key.upper()}")
        print(f"-> Phiên bản        : {ACTIVE_MODEL_INFO['version']} (AUC: {ACTIVE_MODEL_INFO['auc_roc']})")

        yield

    except Exception as e:
        print(f"❌ Lỗi khởi động hệ thống: {e}")
        raise e

    finally:
        print("🛑 Đang tắt hệ thống và giải phóng RAM...")


# ==================================================
# 3. KHỞI TẠO FASTAPI APP
# ==================================================
app = FastAPI(
    title="Real-time Credit Scoring API",
    description="API chấm điểm tín dụng với XGBoost và XAI (SHAP)",
    version="1.0",
    lifespan=lifespan
)


class CustomerData(BaseModel):
    features: dict


def format_value(val):
    if isinstance(val, (int, float, np.number)):
        return round(float(val), 2)
    return str(val)


# ==================================================
# 4. ENDPOINT CHÍNH: /predict
# ==================================================
@app.post("/predict")
async def predict_risk(data: CustomerData):
    try:
        input_features = data.features

        missing_cols = [col for col in REQUIRED_FEATURES if col not in input_features]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Thiếu dữ liệu các trường: {missing_cols}")

        df_input = pd.DataFrame([input_features])[REQUIRED_FEATURES]
        for col in df_input.select_dtypes(include=['object']).columns:
            df_input[col] = df_input[col].astype('category')
        #Xac suat vo no
        prob = float(MODEL.predict_proba(df_input)[0, 1])

        explanation = []
        if EXPLAINER is not None:
            shap_values = EXPLAINER.shap_values(df_input)[0]
            for feat, shap_val, actual_val in zip(REQUIRED_FEATURES, shap_values, df_input.iloc[0]):
                explanation.append({
                    "feature": feat,
                    "value": format_value(actual_val),
                    "impact": float(shap_val)
                })
            explanation.sort(key=lambda x: x["impact"], reverse=True)
            top_red_flags = explanation[:3]
            top_green_flags = explanation[-2:]
        else:
            top_red_flags, top_green_flags = [], []

        threshold = CONFIG["inference_settings"]["threshold_high_risk"]
        decision = "REJECT" if prob > threshold else "REVIEW/APPROVE"

        return {
            "status": "success",
            "model_metadata": {
                "active_model": CONFIG["active_model"],
                "version": ACTIVE_MODEL_INFO["version"],
                "train_date": CONFIG["train_date"]
            },
            "prediction": {
                "default_probability": round(prob, 4),
                "decision": decision
            },
            "explanations": {
                "top_red_flags": top_red_flags,
                "top_green_flags": top_green_flags
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)