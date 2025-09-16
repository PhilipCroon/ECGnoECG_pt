import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.parametrize("filename, content_type", [
    ("test_ecg.png", "image/png"),
    ("test_ecg.jpeg", "image/jpeg"),
    ("test_ecg.jpg", "image/jpeg"),
])

def test_predict_image_various_formats(filename, content_type):
    api_key = os.getenv("API_KEY", "supersecret")
    headers = {"X-API-Key": api_key}
    test_file = os.path.join(os.path.dirname(__file__), "data", filename)
    with open(test_file, "rb") as f:
        response = client.post("/predict", files={"file": (filename, f, content_type)}, headers=headers)
    assert response.status_code == 200
    result = response.json()
    assert "is_ecg" in result
    assert "probability" in result

def test_predict_pdf():
    api_key = os.getenv("API_KEY", "supersecret")
    headers = {"X-API-Key": api_key}
    test_file = os.path.join(os.path.dirname(__file__), "data", "test_ecg.pdf")
    if not os.path.exists(test_file):
        pytest.skip("PDF test file not available")
    with open(test_file, "rb") as f:
        response = client.post("/predict", files={"file": ("test_ecg.pdf", f, "application/pdf")}, headers=headers)
    assert response.status_code == 200
    result = response.json()
    assert "is_ecg" in result
    assert "probability" in result

def test_predict_dicom():
    api_key = os.getenv("API_KEY", "supersecret")
    headers = {"X-API-Key": api_key}
    test_file = os.path.join(os.path.dirname(__file__), "data", "test_ecg.dcm")
    with open(test_file, "rb") as f:
        response = client.post("/predict", files={"file": ("test_ecg.dcm", f, "application/octet-stream")}, headers=headers)
    assert response.status_code == 200
    result = response.json()
    assert "is_ecg" in result
    assert "probability" in result

def test_predict_async_image():
    api_key = os.getenv("API_KEY", "supersecret")
    headers = {"X-API-Key": api_key}
    test_file = os.path.join(os.path.dirname(__file__), "data", "test_ecg.png")
    with open(test_file, "rb") as f:
        response = client.post("/predict_async", files={"file": ("test_ecg.png", f, "image/png")}, headers=headers)
    assert response.status_code == 200
    result = response.json()
    assert "task_id" in result

def test_predict_async_dicom():
    api_key = os.getenv("API_KEY", "supersecret")
    headers = {"X-API-Key": api_key}
    test_file = os.path.join(os.path.dirname(__file__), "data", "test_ecg.dcm")
    with open(test_file, "rb") as f:
        response = client.post("/predict_async", files={"file": ("test_ecg.dcm", f, "application/octet-stream")}, headers=headers)
    assert response.status_code == 200
    result = response.json()
    assert "task_id" in result

def test_result_pending():
    fake_task_id = "nonexistent-task"
    response = client.get(f"/result/{fake_task_id}")
    assert response.status_code == 200
    result = response.json()
    assert "status" in result
    assert result["status"].lower() in ["pending", "not found", "queued", "processing"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_without_api_key():
    test_file = os.path.join(os.path.dirname(__file__), "data", "test_ecg.png")
    with open(test_file, "rb") as f:
        # Remove any Authorization header or api key query param if used
        response = client.post("/predict", files={"file": ("test_ecg.png", f, "image/png")})
    assert response.status_code == 401

def test_predict_with_api_key():
    api_key = os.getenv("API_KEY", "supersecret")
    headers = {"X-API-Key": api_key}
    test_file = os.path.join(os.path.dirname(__file__), "data", "test_ecg.png")
    with open(test_file, "rb") as f:
        response = client.post("/predict", files={"file": ("test_ecg.png", f, "image/png")}, headers=headers)
    assert response.status_code == 200
    result = response.json()
    assert "is_ecg" in result
    assert "probability" in result