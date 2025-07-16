# %%
import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
from pathlib import Path
from sklearn.metrics import roc_curve
import timm
import torch.nn as nn

# ✅ CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 224
data_root = "/home/pmc57/ECG_preprosessing/data"
val_dir = os.path.join(data_root, "val")
test_dir = os.path.join(data_root, "test_small")

# ✅ TRANSFORM
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# ✅ LOAD VALIDATION DATA FOR THRESHOLDING
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ✅ LOAD TEST DATA (SHUFFLED)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
indices = list(range(len(test_dataset)))
random.shuffle(indices)
shuffled_test_dataset = Subset(test_dataset, indices)
test_loader = DataLoader(shuffled_test_dataset, batch_size=1, shuffle=False)

# ✅ MODEL DEFINITION
model = timm.create_model("efficientnet_lite0", pretrained=False, num_classes=1)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier.in_features, 1),
    nn.Sigmoid()
)

# ✅ LOAD MOST RECENT CHECKPOINT
model_path = sorted(Path("/home/pmc57/ECGnoECG_PT/models").glob("efficientnet_lite0_*/best_model.pth"))[-1]
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f"✅ Loaded model from: {model_path}")

# ✅ CALCULATE OPTIMAL THRESHOLD FROM VALIDATION SET
from sklearn.metrics import roc_curve

all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = outputs.cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
j_scores = tpr - fpr
best_thresh = thresholds[j_scores.argmax()]
print(f"📈 Optimal threshold from validation: {best_thresh:.4f}")


# ✅ PREDICT & SHOW TEST_SMALL EXAMPLES
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        probs = outputs.cpu().numpy().flatten()

        # Plot ECG
        img = images[0].cpu().permute(1, 2, 0).numpy()
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')

        label_text = "ECG" if probs[0] < best_thresh else "no ECG"
        plt.text(5, 20, label_text, fontsize=16,
                 color='red' if label_text == "ECG" else 'green', weight='bold')

        plt.show()
        input("Press Enter to continue...")