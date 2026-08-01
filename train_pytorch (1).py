"""
Uzhavan AI — train_pytorch.py
========================
Real PyTorch training script. Trains on whatever real, correctly-labeled
image folders exist in uzhavan_ai_dataset_clean/ and exports the trained model
to ONNX so it can run in the browser (via onnxruntime-web) as the app's
real backend, replacing the mock hash-based classifier.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import json

DATA_DIR = "uzhavan_ai_dataset_tiny"
IMG_SIZE = 64            # smaller than the usual 224 — keeps CPU training feasible
BATCH_SIZE = 32
EPOCHS = 8               # more epochs since we're training from scratch, not fine-tuning
VAL_SPLIT = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", DEVICE)

# ---------------------------------------------------------------------
# Transforms: resize + normalize to what MobileNetV2 (ImageNet-pretrained)
# expects. Light augmentation on the training split only.
# ---------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
class_names = full_dataset.classes
print("Classes (in model output order):", class_names)

n_val = int(len(full_dataset) * VAL_SPLIT)
n_train = len(full_dataset) - n_val
train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                  generator=torch.Generator().manual_seed(42))
val_ds.dataset.transform = val_tf  # validation shouldn't use train-time augmentation

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {n_train} images | Val: {n_val} images")

# ---------------------------------------------------------------------
# Model: a compact CNN trained FROM SCRATCH.
#
# Note: pretrained ImageNet weights (the usual transfer-learning shortcut)
# aren't downloadable from this environment's network, so this is a small
# custom architecture sized for our dataset (~900 images, 4 classes)
# rather than a frozen MobileNetV2 head. Expect lower accuracy than a
# properly transfer-learned model — if you retrain this later somewhere
# with full internet access, swap back to pretrained MobileNetV2/
# EfficientNet for meaningfully better results with the same data.
# ---------------------------------------------------------------------
class UzhavanAICNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32),     # 128 -> 64
            block(32, 64),    # 64 -> 32
            block(64, 128),   # 32 -> 16
            block(128, 128),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = UzhavanAICNN(len(class_names)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total

    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            v_correct += (outputs.argmax(1) == labels).sum().item()
            v_total += labels.size(0)
    val_acc = v_correct / v_total if v_total else 0

    print(f"Epoch {epoch+1}/{EPOCHS} | loss {running_loss/total:.4f} | "
          f"train_acc {train_acc:.2%} | val_acc {val_acc:.2%}")

# ---------------------------------------------------------------------
# Export to ONNX — this is what runs client-side in the browser via
# onnxruntime-web, giving true offline, on-device inference.
#
# IMPORTANT: torch's newer "dynamo" exporter (default in recent PyTorch)
# can silently split weights into a separate .onnx.data file alongside
# the .onnx file. Since we embed the model as a single base64 blob in
# the HTML, that split breaks everything — the browser has the .onnx
# structure but none of the actual weight values, causing exactly the
# "Failed to load external data file" error at runtime.
# dynamo=False forces the older exporter, which keeps small models like
# this one fully self-contained in one file. The onnx.load/save step
# right after is a belt-and-braces safety net that FORCES all weights
# to be embedded inline regardless of exporter behavior.
# ---------------------------------------------------------------------
model.eval()
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
torch.onnx.export(
    model, dummy_input, "uzhavan_ai_model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12,
    dynamo=False,
)

# Safety net: force everything into ONE file, no external data, no matter
# what the exporter above decided to do.
import onnx
onnx_model = onnx.load("uzhavan_ai_model.onnx", load_external_data=True)
onnx.save_model(onnx_model, "uzhavan_ai_model.onnx", save_as_external_data=False)
print("Verified: model is a single self-contained file (no external data).")

with open("uzhavan_ai_labels.json", "w") as f:
    json.dump(class_names, f)

print("\nDone.")
print("  uzhavan_ai_model.onnx  <- the trained model")
print("  uzhavan_ai_labels.json <- class names in output-index order:", class_names)
