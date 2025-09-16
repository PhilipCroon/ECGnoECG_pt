import requests
import os
from celery import Celery

celery_app = Celery(
    "ecg_tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
)

@celery_app.task
def run_prediction(file_bytes: bytes, callback_url=None):
    from app.model import get_model
    from app.utils import pil_from_bytes

    pil = pil_from_bytes(file_bytes)
    model = get_model()
    prob = model.predict_prob(pil)
    thr = model.threshold()
    result = {"prob": prob, "is_ecg": bool(prob < thr)}
    if callback_url is not None:
        try:
            secret = os.getenv("CALLBACK_SECRET")
            headers = {"Authorization": f"Bearer {secret}"} if secret else {}
            requests.post(callback_url, json=result, headers=headers, timeout=10)
        except Exception as e:
            print(f"Error sending callback to {callback_url}: {e}")
    return result