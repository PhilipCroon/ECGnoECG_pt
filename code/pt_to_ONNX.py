# %%
import torch
import timm
import torch.nn as nn
from pathlib import Path

# Rebuild model architecture
model = timm.create_model("efficientnet_lite0", pretrained=False, num_classes=1)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier.in_features, 1),
    nn.Sigmoid()
)
# Best model
model_path = sorted(Path("/home/pmc57/ECGnoECG_PT/models").glob("efficientnet_lite0_*/best_model.pth"))[-1]

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Dummy input for tracing
dummy_input = torch.randn(1, 3, 224, 224)

onnx_path = Path("/home/pmc57/ECGnoECG_PT/models/efficientnet_lite0.onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print(f"✅ Exported to ONNX at: {onnx_path}")
