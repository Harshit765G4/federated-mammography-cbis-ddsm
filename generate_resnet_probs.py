import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import os

PROCESSED_DIR = r"D:\Breast_Cancer_DICOM\processed"
WEIGHTS_DIR = r"D:\Breast_Cancer_DICOM\outputs\weights_curated"
LOGS_DIR = r"D:\Breast_Cancer_DICOM\outputs\logs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
BATCH_SIZE = 8
DROPOUT_P = 0.5
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
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(model.fc.in_features, 2))
    return model.to(DEVICE)


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


val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_labels_curated.csv"))

for fold in range(N_FOLDS):
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    val_ds = MammoDataset(val_df, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model()
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, f"resnet50_fold{fold}.pt")))

    probs, labels = evaluate_with_tta(model, val_loader)

    val_df_out = val_df.copy()
    val_df_out["resnet_prob"] = probs
    out_path = os.path.join(LOGS_DIR, f"resnet_fold{fold}_val_probs.csv")
    val_df_out.to_csv(out_path, index=False)
    print(f"Fold {fold}: saved {len(val_df_out)} rows -> {out_path}")

print("\nDone.")