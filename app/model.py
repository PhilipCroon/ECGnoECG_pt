# app/model.py
import os, pathlib
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image

MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS")
if not MODEL_WEIGHTS or not os.path.exists(MODEL_WEIGHTS):
    candidates = sorted(pathlib.Path("/app/models").rglob("best_model.pth"))
    if not candidates:
        raise FileNotFoundError("No best_model.pth found in /app/models")
    MODEL_WEIGHTS = str(candidates[-1])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = int(os.getenv("IMG_SIZE", 224))
MODEL_NAME = os.getenv("MODEL_NAME", "efficientnet_lite0")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  # fallback if no val threshold provided

# same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def build_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
    # handle efficientnet-lite head
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
    return model

class ECGModel:
    def __init__(self):
        self.model = build_model()
        state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model.to(DEVICE)
        self.model.eval()

    @torch.inference_mode()
    def predict_prob(self, pil_image: Image.Image) -> float:
        x = transform(pil_image.convert("RGB"))
        x = x.unsqueeze(0).to(DEVICE)
        prob = self.model(x).sigmoid().item()
        return float(prob)

    def threshold(self) -> float:
        return THRESHOLD

model_singleton: ECGModel | None = None

def get_model() -> ECGModel:
    global model_singleton
    if model_singleton is None:
        model_singleton = ECGModel()
    return model_singleton