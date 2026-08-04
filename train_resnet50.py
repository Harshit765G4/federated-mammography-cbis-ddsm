import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import os
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from tqdm import tqdm

PROCESSED_DIR = r"D:\Breast_Cancer_DICOM\processed"
WEIGHTS_DIR = r"D:\Breast_Cancer_DICOM\outputs\weights"
LOGS_DIR = r"D:\Breast_Cancer_DICOM\outputs\logs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
IMG_SIZE = 512
WEIGHT_DECAY = 1e-5
DROPOUT_P = 0.5
EARLY_STOP_PATIENCE = 6


class MammoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["clahe_image_path"], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.stack([img, img, img], axis=-1)
        label = 1 if row["pathology"] == "MALIGNANT" else 0

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("Using device:", DEVICE)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_labels_final.csv"))

    train_df = df[df["fold"] != 0].reset_index(drop=True)
    val_df = df[df["fold"] == 0].reset_index(drop=True)

    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")
    print("Train label balance:\n", train_df["pathology"].apply(lambda p: "MALIGNANT" if p == "MALIGNANT" else "BENIGN").value_counts())
    print("Val label balance:\n", val_df["pathology"].apply(lambda p: "MALIGNANT" if p == "MALIGNANT" else "BENIGN").value_counts())

    train_ds = MammoDataset(train_df, transform=train_transform)
    val_ds = MammoDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(model.fc.in_features, 2)
    )

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    class_counts = train_df["pathology"].apply(lambda p: 1 if p == "MALIGNANT" else 0).value_counts()
    n_benign = class_counts[0]
    n_malignant = class_counts[1]
    weight = torch.tensor([1.0, n_benign / n_malignant], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)
    print(f"Class weights (benign, malignant): [1.0, {n_benign/n_malignant:.3f}]")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0.0
    epochs_no_improve = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_auc = roc_auc_score(all_labels, all_probs)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, val_auc={val_auc:.4f}, lr={current_lr:.2e}")
        history.append({"epoch": epoch+1, "train_loss": train_loss, "val_acc": val_acc, "val_auc": val_auc, "lr": current_lr})

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "resnet50_best.pt"))
            print(f"  -> New best model saved (AUC={val_auc:.4f})")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "resnet50_best.pt")))
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("\n=== FINAL BEST MODEL METRICS (ResNet50) ===")
    print("Confusion matrix:\n", cm)
    print(f"Accuracy:    {accuracy_score(all_labels, all_preds):.4f}")
    print(f"AUC:         {best_auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    pd.DataFrame(history).to_csv(os.path.join(LOGS_DIR, "resnet50_training_history.csv"), index=False)
    print(f"\nSaved training history: {os.path.join(LOGS_DIR, 'resnet50_training_history.csv')}")


if __name__ == "__main__":
    main()