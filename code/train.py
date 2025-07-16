# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
import timm
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

print('Start')

# ✅ 1. CONFIGURATION
data_dir = "/home/pmc57/ECG_preprosessing/data"
batch_size = 32
img_size = 224
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Config done, device = {device}')

# ✅ 2. TRANSFORMS
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

val_test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

print('Transformer done')

# ✅ 3. DATASETS
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transform)
test_small_dataset = datasets.ImageFolder(os.path.join(data_dir, "test_small"), transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_small_loader = DataLoader(test_small_dataset, batch_size=batch_size, shuffle=False)

print('Datasets done')

# ✅ 4. LOAD MODEL FROM timm
model = timm.create_model("efficientnet_lite0", pretrained=True, num_classes=1)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier.in_features, 1),
    nn.Sigmoid()
)
model = model.to(device)

print(f'Model done - {model}')

# ✅ 5. LOSS & OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

print(f'Criterion done')

# ✅ Output directory
now = datetime.now().strftime("%Y%m%d_%H%M")
model_name = "efficientnet_lite0"
output_dir = Path(f"/home/pmc57/ECGnoECG_PT/models/{model_name}_{now}")
output_dir.mkdir(parents=True, exist_ok=True)

# ✅ 7. EVALUATION FUNCTION
def evaluate(loader, name="Set"):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    auc = roc_auc_score(targets, preds)
    print(f"{name} AUC: {auc:.4f}")
    return auc

#%% ✅ Save training settings
with open(output_dir / "config.txt", "w") as f:
    f.write(f"Model: {model_name}\n")
    f.write(f"Epochs: {num_epochs}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Image size: {img_size}\n")
    f.write(f"Optimizer: Adam\n")
    f.write(f"LR: 1e-4, Scheduler: ExponentialLR gamma=0.9\n")

log_path = output_dir / "train_log.txt"

best_auc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item(), acc=correct / total)

    scheduler.step()
    accuracy = correct / total
    avg_loss = total_loss / total

    val_auc = evaluate(val_loader, "Validation")
    # Save model
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), output_dir / "best_model.pth")
        
        # Save log
    with open(log_path, "a") as log_file:
        log_file.write(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.4f}, val_auc={val_auc:.4f}\n")
    
    print(f"✅ Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, Val AUC={val_auc:.4f}")
    
# %%
print("\n📊 Final Evaluation:")

#%% ✅ Find latest model folder if output_dir not manually set
if 'output_dir' not in locals() or not output_dir.exists():
    base_dir = Path("/home/pmc57/ECGnoECG_PT/models")
    model_folders = sorted(base_dir.glob("efficientnet_lite0_*"), reverse=True)
    if not model_folders:
        raise FileNotFoundError("❌ No saved model folders found!")
    output_dir = model_folders[0]
    print(f"📂 Using latest model folder: {output_dir}")

    # ✅ Load a saved model
checkpoint_path = output_dir / "best_model.pth"  # or "final_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print(f"\n✅ Loaded model from: {checkpoint_path}")

val_auc = evaluate(val_loader, "Validation")
test_auc = evaluate(test_loader, "Test")
test_small_auc = evaluate(test_small_loader, "Test Small")

# Optionally write to log
with open(log_path, "a") as log_file:
    log_file.write("\nFinal Results:\n")
    log_file.write(f"Validation AUC: {val_auc:.4f}\n")
    log_file.write(f"Test AUC: {test_auc:.4f}\n")
    log_file.write(f"Test Small AUC: {test_small_auc:.4f}\n")

# %%
# calculate optimal threshold

# from sklearn.metrics import roc_curve

# # Collect validation outputs and labels
# model.eval()
# all_probs = []
# all_labels = []

# with torch.no_grad():
#     for images, labels in val_loader:
#         images = images.to(device)
#         outputs = model(images)
#         probs = torch.sigmoid(outputs).cpu().numpy().flatten()
#         all_probs.extend(probs)
#         all_labels.extend(labels.numpy())

# # Compute ROC and best threshold
# fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
# j_scores = tpr - fpr
# best_thresh = thresholds[j_scores.argmax()]
# print(f"📈 Optimal threshold: {best_thresh:.4f}")


# %%

from torch.utils.data import Subset
import random

# ✅ Shuffle test_small_dataset
indices = list(range(len(test_small_dataset)))
random.shuffle(indices)
shuffled_test_small_dataset = Subset(test_small_dataset, indices)
test_small_loader = DataLoader(shuffled_test_small_dataset, batch_size=1, shuffle=False)

# ✅ Set your threshold
threshold = 0.6847

# ✅ Loop over shuffled test_small_loader and visualize predictions
model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(test_small_loader):
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs >= threshold).astype(int)

        # Show image with text overlay
        img = images[0].cpu().permute(1, 2, 0).numpy()
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')

        if probs[0] < threshold:
            label_text = "ECG"
        else:
            label_text = "no ECG"

        # Add text in top-left corner
        plt.text(5, 20, label_text, fontsize=16, color='red' if 'ECG' in label_text else 'green', weight='bold')

        plt.show()
        input("Press Enter to continue...")

# %%
from sklearn.metrics import roc_curve

all_probs = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
j_scores = tpr - fpr
best_thresh = thresholds[j_scores.argmax()]
print(f"📈 Optimal threshold from validation set: {best_thresh:.4f}")
# %%
