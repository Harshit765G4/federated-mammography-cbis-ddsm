import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import os
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from tqdm import tqdm

PROCESSED_DIR = r"D:\Breast_Cancer_DICOM\processed"
WEIGHTS_DIR = r"D:\Breast_Cancer_DICOM\outputs\weights_densenet_curated"
LOGS_DIR = r"D:\Breast_Cancer_DICOM\outputs\logs"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 512
WEIGHT_DECAY = 1e-5
DROPOUT_P = 0.5
EARLY_STOP_PATIENCE = 6
N_FOLDS = 5


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


def build_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(model.classifier.in_features, 2))

    for param in model.parameters():
        param.requires_grad = False
    # DenseNet121's deepest blocks: denseblock4 + norm5 (final norm layer), analogous to ResNet's layer3/layer4
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.features.transition3.parameters():
        param.requires_grad = True
    for param in model.features.norm5.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


def optimal_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def evaluate_with_tta(model, val_loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            probs_orig = torch.softmax(model(imgs), dim=1)[:, 1]
            imgs_flipped = torch.flip(imgs, dims=[3])
            probs_flip = torch.softmax(model(imgs_flipped), dim=1)[:, 1]
            probs_avg = ((probs_orig + probs_flip) / 2).cpu().numpy()
            all_probs.extend(probs_avg)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
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

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_labels_curated.csv"))
    print(f"Curated dataset size: {len(df)} rows, {df['patient_id'].nunique()} patients")

    fold_results = []

    for fold in range(N_FOLDS):
        print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")

        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)

        train_ds = MammoDataset(train_df, transform=train_transform)
        val_ds = MammoDataset(val_df, transform=val_transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = build_model()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if fold == 0:
            print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        class_counts = train_df["pathology"].apply(lambda p: 1 if p == "MALIGNANT" else 0).value_counts()
        weight = torch.tensor([1.0, class_counts[0] / class_counts[1]], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc = 0.0
        epochs_no_improve = 0
        ckpt_path = os.path.join(WEIGHTS_DIR, f"densenet121_fold{fold}.pt")

        for epoch in range(EPOCHS):
            model.train()
            for imgs, labels in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{EPOCHS}"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            probs, labels_np = [], []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs = imgs.to(DEVICE)
                    p = torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
                    probs.extend(p)
                    labels_np.extend(lbls.numpy())
            val_auc = roc_auc_score(labels_np, probs)

            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch+1}, best AUC={best_auc:.4f}")
                break

        model.load_state_dict(torch.load(ckpt_path))
        tta_probs, tta_labels = evaluate_with_tta(model, val_loader)

        thresh = optimal_threshold(tta_labels, tta_probs)
        tta_preds = (tta_probs > thresh).astype(int)

        acc = accuracy_score(tta_labels, tta_preds)
        auc = roc_auc_score(tta_labels, tta_probs)
        cm = confusion_matrix(tta_labels, tta_preds)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)

        print(f"Fold {fold} FINAL (TTA + threshold={thresh:.3f}): acc={acc:.4f}, auc={auc:.4f}, sens={sens:.4f}, spec={spec:.4f}")
        fold_results.append({"fold": fold, "threshold": thresh, "accuracy": acc, "auc": auc, "sensitivity": sens, "specificity": spec})

        # save probs for later ensembling with ResNet50
        val_df_out = val_df.copy()
        val_df_out["densenet_prob"] = tta_probs
        val_df_out.to_csv(os.path.join(LOGS_DIR, f"densenet_fold{fold}_val_probs.csv"), index=False)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(LOGS_DIR, "densenet121_curated_5fold_results.csv"), index=False)

    print(f"\n{'='*60}\nFINAL DENSENET121 CURATED 5-FOLD SUMMARY (mean ± std)\n{'='*60}")
    for metric in ["accuracy", "auc", "sensitivity", "specificity"]:
        vals = results_df[metric]
        print(f"{metric}: {vals.mean():.4f} ± {vals.std():.4f}")


if __name__ == "__main__":
    main()