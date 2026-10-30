import pytest
from fastapi.testclient import TestClient
from api.main import app

SAMPLE_VALID_FEATURES = {
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


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["active_model"] == "xgboost_core"
    assert data["features_loaded"] == 28


def test_predict_success(client):
    payload = {"features": SAMPLE_VALID_FEATURES}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "default_probability" in data["prediction"]
    prob = data["prediction"]["default_probability"]
    assert 0.0 <= prob <= 1.0
    assert data["prediction"]["decision"] in ["REJECT", "REVIEW/APPROVE"]

    assert "top_red_flags" in data["explanations"]
    assert "top_green_flags" in data["explanations"]
    assert len(data["explanations"]["top_red_flags"]) <= 3
    assert len(data["explanations"]["top_green_flags"]) <= 2


def test_predict_missing_features(client):
    payload = {"features": {"age": 30.0, "credamount_770A": 50000.0}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Thiếu dữ liệu các trường bắt buộc" in response.json()["detail"]


def test_predict_empty_payload(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
