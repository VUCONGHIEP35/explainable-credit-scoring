import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import json
import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import shap
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# ==================================================
# 1. CẤU HÌNH LOGGING CHUẨN MỰC
# ==================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("credit_scoring_api")

# ==================================================
# 2. KHAI BÁO BIẾN TOÀN CỤC & TÍNH TOÁN ĐƯỜNG DẪN
# ==================================================
MODEL = None
EXPLAINER = None
CONFIG = None
REQUIRED_FEATURES = None
ACTIVE_MODEL_INFO = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==================================================
# 3. LIFESPAN: QUẢN LÝ KHỞI ĐỘNG VÀ TẮT SERVER
# ==================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, EXPLAINER, CONFIG, REQUIRED_FEATURES, ACTIVE_MODEL_INFO
    logger.info("Đang khởi động hệ thống và tải Model Artifacts...")

    try:
        config_path = os.path.join(BASE_DIR, "model_artifacts", "config.json")
        features_path = os.path.join(BASE_DIR, "model_artifacts", "selected_features.json")

        # 1. Đọc Metadata Configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy config tại: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            CONFIG = json.load(f)

        # 2. Đọc danh sách 28 Features Vàng
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Không tìm thấy danh sách features tại: {features_path}")
        with open(features_path, "r", encoding="utf-8") as f:
            REQUIRED_FEATURES = json.load(f)

        # 3. Đọc đúng Mô hình đang Active từ Config
        active_model_key = CONFIG["active_model"]
        ACTIVE_MODEL_INFO = CONFIG["models"][active_model_key]

        raw_model_path = ACTIVE_MODEL_INFO["file_path"]
        model_abs_path = os.path.normpath(os.path.join(BASE_DIR, raw_model_path))

        # 4. Tải Mô hình và Khởi tạo SHAP
        if "xgboost" in active_model_key.lower():
            MODEL = xgb.XGBClassifier(enable_categorical=True)
            MODEL.load_model(model_abs_path)
            EXPLAINER = shap.TreeExplainer(MODEL)
        else:
            import joblib
            MODEL = joblib.load(model_abs_path)
            EXPLAINER = None

        logger.info("HỆ THỐNG ĐÃ SẴN SÀNG!")
        logger.info(f"-> Active Model: {active_model_key.upper()}")
        logger.info(f"-> Phiên bản   : {ACTIVE_MODEL_INFO['version']} (AUC: {ACTIVE_MODEL_INFO['auc_roc']})")

        yield

    except Exception as e:
        logger.error(f"Lỗi khởi động hệ thống: {e}", exc_info=True)
        raise e

    finally:
        logger.info("Đang tắt hệ thống và giải phóng RAM...")


# ==================================================
# 4. KHỞI TẠO FASTAPI APP
# ==================================================
app = FastAPI(
    title="Explainable Credit Scoring API",
    description=(
        "Hệ thống chấm điểm rủi ro tín dụng thời gian thực tích hợp Explainable AI (TreeSHAP). "
        "Hỗ trợ phân tích Red Flags (tăng rủi ro) và Green Flags (giảm rủi ro) cho từng hồ sơ vay."
    ),
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "VU CONG HIEP",
        "url": "https://github.com/VUCONGHIEP35/explainable-credit-scoring"
    }
)


# ==================================================
# 5. DATA SCHEMAS
# ==================================================
class CustomerData(BaseModel):
    features: dict = Field(
        ...,
        description="Dictionary chứa đúng 28 biến đầu vào theo chuẩn hệ thống"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
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
            }
        }
    }


def format_value(val):
    if isinstance(val, (int, float, np.number)):
        return round(float(val), 2)
    return str(val)


# ==================================================
# 6. ENDPOINTS
# ==================================================
@app.get("/", tags=["General"])
async def root():
    """Thông tin tổng quan dịch vụ và đường dẫn tài liệu API."""
    return {
        "service": "Explainable Credit Scoring API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Kiểm tra sức khỏe hệ thống và tình trạng nạp mô hình vào RAM."""
    is_ready = MODEL is not None and CONFIG is not None and REQUIRED_FEATURES is not None
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống chưa nạp xong Model Artifacts."
        )
    return {
        "status": "healthy",
        "active_model": CONFIG.get("active_model"),
        "version": ACTIVE_MODEL_INFO.get("version"),
        "features_loaded": len(REQUIRED_FEATURES)
    }


@app.post("/predict", tags=["Inference"])
async def predict_risk(data: CustomerData):
    """
    Dự báo xác suất vỡ nợ (PD) và giải thích quyết định bằng SHAP:
    - Trả về xác suất vỡ nợ (0.0 đến 1.0)
    - Quyết định: REJECT hoặc REVIEW/APPROVE dựa trên ngưỡng rủi ro
    - Top Red Flags (các biến làm tăng rủi ro cao nhất)
    - Top Green Flags (các biến làm giảm rủi ro tốt nhất)
    """
    try:
        input_features = data.features

        missing_cols = [col for col in REQUIRED_FEATURES if col not in input_features]
        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Thiếu dữ liệu các trường bắt buộc: {missing_cols}"
            )

        df_input = pd.DataFrame([input_features])[REQUIRED_FEATURES]
        for col in df_input.select_dtypes(include=['object']).columns:
            df_input[col] = df_input[col].astype('category')

        # Xác suất vỡ nợ (Probability of Default - PD)
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi xử lý dự báo: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
