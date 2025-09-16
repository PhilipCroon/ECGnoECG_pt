# app/main.py
import base64
import io
import os
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.model import get_model
from app.utils import pil_from_bytes, dicom_to_pil, pdf_to_pil
from ecg.ecg import ECG, process_ecg_plot_from_signal  # your ecg module

from PIL import Image
from pdf2image import convert_from_bytes

from app.queue import run_prediction, celery_app

app = FastAPI(title="ECG Detector Service", version="1.2")

# Allow CORS for all origins (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    expected_api_key = os.getenv("API_KEY", "supersecret")
    if x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/health")
def health():
    return {"status": "ok"}

class PredictionResponse(BaseModel):
    input_type: str
    probability: float
    threshold: float
    is_ecg: bool
    patient_id: str | None = None
    accession_number: str | None = None
    render_png_b64: str | None = None

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(file: UploadFile = File(...), return_render: bool = False):
    content = await file.read()
    content_type = (file.content_type or "").lower()
    in_type = "unknown"
    pil = None
    patient_id = None
    accession_number = None

    # --- Image ---
    if content_type in {"image/png", "image/jpeg", "image/jpg"}:
        pil = pil_from_bytes(content)
        in_type = "image"

    # --- PDF (first page only) ---
    elif content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(content, dpi=300)
            if not pages:
                raise HTTPException(status_code=400, detail="No pages found in PDF")
            pil = pages[0].convert("RGB")  # first page only
            in_type = "pdf"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to convert PDF: {e}")

    # --- DICOM ---
    elif content_type in {
        "application/dicom", "application/dicom+json", "application/octet-stream"
    } or file.filename.lower().endswith((".dcm", ".dicom")):
        try:
            tmp_path = f"/tmp/{file.filename}"
            with open(tmp_path, "wb") as f:
                f.write(content)

            ecg = ECG(tmp_path, info_print=False)
            signal = ecg.signals
            accession_number = getattr(ecg, "accession", None)
            patient_id = getattr(ecg, "patient_id", None)

            plot_path = tmp_path.replace(".dcm", "_plot.png")
            process_ecg_plot_from_signal(signal, accession_number or "unknown", plot_path)
            pil = Image.open(plot_path).convert("RGB")
            in_type = "dicom"
        except Exception as e:
            # Fallback: pixel-based DICOM
            try:
                pil, in_type = dicom_to_pil(content)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Failed to process DICOM: {e}")

    else:
        # Fallback: try image decode
        try:
            pil = pil_from_bytes(content)
            in_type = "image"
        except Exception:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    # --- Run model ---
    model = get_model()
    prob = model.predict_prob(pil)
    thr = model.threshold()
    is_ecg = bool(prob < thr)

    # --- Build response ---
    resp = {
        "input_type": in_type,
        "probability": prob,
        "threshold": thr,
        "is_ecg": is_ecg,
        "patient_id": patient_id,
        "accession_number": accession_number,
    }

    if return_render and pil is not None:
        buff = io.BytesIO()
        pil.save(buff, format="PNG")
        resp["render_png_b64"] = base64.b64encode(buff.getvalue()).decode("ascii")

    return JSONResponse(resp)

@app.post("/predict_async", dependencies=[Depends(verify_api_key)])
async def predict_async(file: UploadFile = File(...), callback_url: str | None = Form(None)):
    content = await file.read()
    task = run_prediction.delay(content, callback_url)
    return {"task_id": task.id}

@app.get("/result/{task_id}", dependencies=[Depends(verify_api_key)])
def get_result(task_id: str):
    result = celery_app.AsyncResult(task_id)
    if result.ready():
        return result.result
    return {"status": "pending"}