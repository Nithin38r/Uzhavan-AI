"""
Uzhavan AI — train_pytorch.py
============================
PyTorch training script combining:
  - Pretrained MobileNetV2 (transfer learning, not from-scratch)
  - A CORRECTLY separated train/val split (no shared-transform bug)
  - Class weighting for imbalance
  - Early stopping + best-checkpoint saving
  - Two-phase training: frozen backbone -> fine-tune last layers
  - ONNX export for onnxruntime-web (browser / PWA inference)

--------------------------------------------------------------------
EXPECTED FOLDER STRUCTURE
--------------------------------------------------------------------
Option A (recommended — matches how you already split for TensorFlow):
    uzhavan_ai_dataset/
        train/
            Tomato_Healthy/*.jpg
            Tomato_Early_Blight/*.jpg
            ...
        val/
            Tomato_Healthy/*.jpg
            ...

Option B (single folder — script will split it for you):
    uzhavan_ai_dataset/
        Tomato_Healthy/*.jpg
        Tomato_Early_Blight/*.jpg
        ...

The script auto-detects which structure you have.

--------------------------------------------------------------------
INSTALL (run once):
--------------------------------------------------------------------
pip install torch torchvision onnx scikit-learn pillow

--------------------------------------------------------------------
RUN:
--------------------------------------------------------------------
python train_pytorch.py
"""

import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = "uzhavan_ai_dataset"   # change to your actual dataset folder
IMG_SIZE = 224                     # MobileNetV2's native pretrained input size
BATCH_SIZE = 16                    # small on purpose — small dataset
VAL_SPLIT = 0.2                    # only used if there's no separate train/val folder

EPOCHS_HEAD = 25                   # phase 1: train classification head only
EPOCHS_FINETUNE = 15               # phase 2: unfreeze + fine-tune
FINE_TUNE_LAYERS = 20              # unfreeze last N params-groups of backbone
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
PATIENCE = 6                       # early stopping patience (epochs with no improvement)

MODEL_OUT_DIR = "model_output"
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", DEVICE)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# STEP 1: DATA — with a properly separated train/val transform
# ============================================================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TransformedSubset(Dataset):
    """
    Wraps a Subset so train/val each get their OWN transform.
    This is the fix for the bug where random_split() shares a single
    underlying dataset object, causing .transform reassignment to
    silently overwrite BOTH splits (i.e. training loses augmentation).
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = self.transform(img)
        return img, label


def build_datasets():
    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        print("Detected separate train/ and val/ folders.")
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        class_names = train_ds.classes
        train_targets = train_ds.targets
    else:
        print("No train/val split found — splitting single folder automatically.")
        # Load with NO transform here; each subset applies its own below.
        raw_ds = datasets.ImageFolder(DATA_ROOT, transform=None)
        class_names = raw_ds.classes

        n_val = int(len(raw_ds) * VAL_SPLIT)
        n_train = len(raw_ds) - n_val
        train_subset, val_subset = random_split(
            raw_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_ds = TransformedSubset(train_subset, train_tf)
        val_ds = TransformedSubset(val_subset, val_tf)

        # Pull labels out for class-weight computation
        train_targets = [raw_ds.targets[i] for i in train_subset.indices]

    return train_ds, val_ds, class_names, train_targets


# ============================================================
# STEP 2: CLASS WEIGHTS (handles imbalance, same idea as the TF version)
# ============================================================
def get_class_weights(train_targets, num_classes):
    class_indices = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced", classes=class_indices, y=train_targets
    )
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ============================================================
# STEP 3: MODEL — pretrained MobileNetV2, frozen backbone + new head
# ============================================================
def build_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze the backbone (feature extractor) for phase 1
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier head — MobileNetV2's default head is 1280 -> 1000 (ImageNet)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes),
    )
    return model.to(DEVICE)


# ============================================================
# STEP 4: TRAIN LOOP (shared by both phases) with early stopping
# ============================================================
def run_training(model, train_loader, val_loader, criterion, optimizer,
                  epochs, phase_name):
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ---- Train ----
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

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validate ----
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * imgs.size(0)
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total += labels.size(0)

        val_loss = v_loss / v_total if v_total else float("inf")
        val_acc = v_correct / v_total if v_total else 0

        print(f"[{phase_name}] Epoch {epoch+1}/{epochs} | "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.2%} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.2%}")

        # ---- Early stopping on val_loss, tracking best weights ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[{phase_name}] Early stopping — no improvement for "
                      f"{PATIENCE} epochs.")
                break

    model.load_state_dict(best_state)  # restore best checkpoint
    return model


# ============================================================
# STEP 5: FINE-TUNE — unfreeze last N layers of the backbone
# ============================================================
def unfreeze_last_layers(model, n_layers):
    backbone_params = list(model.features.parameters())
    for param in backbone_params[-n_layers:]:
        param.requires_grad = True


# ============================================================
# STEP 6: EXPORT TO ONNX (for onnxruntime-web / browser inference)
#
# IMPORTANT: torch's newer "dynamo" exporter (default in recent PyTorch)
# can silently split weights into a separate .onnx.data file alongside
# the .onnx file. If you embed the model as a single base64 blob in the
# app's HTML, that split breaks everything — the browser has the .onnx
# structure but none of the actual weight values, causing a
# "Failed to load external data file" error at runtime.
# dynamo=False forces the older exporter, which keeps small models like
# this one fully self-contained in one file. The onnx.load/save step
# right after is a belt-and-braces safety net that FORCES all weights
# to be embedded inline regardless of exporter behavior.
# ============================================================
def export_onnx(model, class_names):
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    onnx_path = os.path.join(MODEL_OUT_DIR, "uzhavan_ai_model.onnx")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
        dynamo=False,
    )

    # Safety net: force everything into ONE file, no external data, no
    # matter what the exporter above decided to do.
    import onnx
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
    print("Verified: model is a single self-contained file (no external data).")

    labels_path = os.path.join(MODEL_OUT_DIR, "uzhavan_ai_labels.json")
    with open(labels_path, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\n✅ ONNX model saved to: {onnx_path}")
    print(f"✅ Label map saved to: {labels_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    train_ds, val_ds, class_names, train_targets = build_datasets()
    num_classes = len(class_names)
    print(f"Classes (model output order): {class_names}")
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    class_weights = get_class_weights(train_targets, num_classes)
    print(f"Class weights: {class_weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Phase 1: train head only ----
    print("\n=== PHASE 1: Training classification head (backbone frozen) ===")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD
    )
    model = run_training(model, train_loader, val_loader, criterion, optimizer,
                          EPOCHS_HEAD, "phase1")

    # ---- Phase 2: fine-tune last backbone layers ----
    print("\n=== PHASE 2: Fine-tuning top backbone layers ===")
    unfreeze_last_layers(model, FINE_TUNE_LAYERS)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINETUNE
    )
    model = run_training(model, train_loader, val_loader, criterion, optimizer,
                          EPOCHS_FINETUNE, "phase2")

    # ---- Final evaluation ----
    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            v_correct += (outputs.argmax(1) == labels).sum().item()
            v_total += labels.size(0)
    print(f"\n=== Final validation accuracy: {v_correct / v_total:.2%} ===")

    # ---- Save + export ----
    torch.save(model.state_dict(), os.path.join(MODEL_OUT_DIR, "uzhavan_ai_model_state.pt"))
    export_onnx(model, class_names)


if __name__ == "__main__":
    main()
