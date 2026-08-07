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
WEIGHTS_DIR = r"D:\Breast_Cancer_DICOM\outputs\weights_500"
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
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(model.fc.in_features, 2))

    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    for module in model.layer3.modules():
        if isinstance(module, nn.BatchNorm2d):
            for p in module.parameters():
                p.requires_grad = True
    for module in model.layer4.modules():
        if isinstance(module, nn.BatchNorm2d):
            for p in module.parameters():
                p.requires_grad = True

    return model.to(DEVICE)


def set_bn_eval_for_frozen(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def optimal_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    return thresholds[np.argmax(j_scores)]


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / denom
    return (max(0, center - margin), min(1, center + margin))


def evaluate_with_tta(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
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

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_labels_500_final.csv"))
    print(f"500-subset size: {len(df)} rows, {df['patient_id'].nunique()} patients")

    leak = df.groupby("patient_id")["fold"].nunique()
    n_leak = (leak > 1).sum()
    print(f"Patients spanning multiple folds: {n_leak} (must be 0)")
    if n_leak > 0:
        raise RuntimeError("Leakage detected -- stop.")

    fixed_results = []
    cal_results = []

    for fold in range(N_FOLDS):
        print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")

        train_df = df[df["fold"] != fold].reset_index(drop=True)
        remainder_df = df[df["fold"] == fold].reset_index(drop=True)

        cal_patients = remainder_df["patient_id"].drop_duplicates().sample(frac=0.5, random_state=42)
        cal_df = remainder_df[remainder_df["patient_id"].isin(cal_patients)].reset_index(drop=True)
        test_df = remainder_df[~remainder_df["patient_id"].isin(cal_patients)].reset_index(drop=True)

        if test_df["pathology"].nunique() < 2 or cal_df["pathology"].nunique() < 2:
            print(f"Skipping fold {fold} -- insufficient class diversity in cal/test split")
            continue

        train_ds = MammoDataset(train_df, transform=train_transform)
        cal_ds = MammoDataset(cal_df, transform=val_transform)
        test_ds = MammoDataset(test_df, transform=val_transform)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        cal_loader = DataLoader(cal_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = build_model()

        class_counts = train_df["pathology"].apply(lambda p: 1 if p == "MALIGNANT" else 0).value_counts()
        weight = torch.tensor([1.0, class_counts.get(0,1) / class_counts.get(1,1)], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc = 0.0
        epochs_no_improve = 0
        ckpt_path = os.path.join(WEIGHTS_DIR, f"resnet50_fold{fold}.pt")

        for epoch in range(EPOCHS):
            model.train()
            set_bn_eval_for_frozen(model)
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
                for imgs, lbls in cal_loader:
                    imgs = imgs.to(DEVICE)
                    p = torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
                    probs.extend(p)
                    labels_np.extend(lbls.numpy())
            cal_auc = roc_auc_score(labels_np, probs)

            if cal_auc > best_auc:
                best_auc = cal_auc
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch+1}, best cal AUC={best_auc:.4f}")
                break

        model.load_state_dict(torch.load(ckpt_path))

        cal_probs, cal_labels = evaluate_with_tta(model, cal_loader)
        thresh = optimal_threshold(cal_labels, cal_probs)

        test_probs, test_labels = evaluate_with_tta(model, test_loader)
        n_test = len(test_labels)
        auc = roc_auc_score(test_labels, test_probs)

        preds_opt = (test_probs > thresh).astype(int)
        cm = confusion_matrix(test_labels, preds_opt)
        tn, fp, fn, tp = cm.ravel()
        acc_opt = accuracy_score(test_labels, preds_opt)
        sens_opt = tp / (tp + fn)
        spec_opt = tn / (tn + fp)
        ci_low, ci_high = wilson_ci(int(acc_opt * n_test), n_test)

        preds_fixed = (test_probs > 0.5).astype(int)
        cm2 = confusion_matrix(test_labels, preds_fixed)
        tn2, fp2, fn2, tp2 = cm2.ravel()
        acc_fixed = accuracy_score(test_labels, preds_fixed)
        sens_fixed = tp2 / (tp2 + fn2)
        spec_fixed = tn2 / (tn2 + fp2)

        print(f"Fold {fold}: AUC={auc:.4f}, n_test={n_test}")
        print(f"  Cal threshold={thresh:.3f}: acc={acc_opt:.4f} [{ci_low:.3f},{ci_high:.3f}], sens={sens_opt:.4f}, spec={spec_opt:.4f}")
        print(f"  Fixed 0.5:              acc={acc_fixed:.4f}, sens={sens_fixed:.4f}, spec={spec_fixed:.4f}")

        cal_results.append({"fold": fold, "auc": auc, "threshold": thresh, "n_test": n_test,
                             "accuracy": acc_opt, "ci_low": ci_low, "ci_high": ci_high,
                             "sensitivity": sens_opt, "specificity": spec_opt})
        fixed_results.append({"fold": fold, "auc": auc, "n_test": n_test,
                               "accuracy": acc_fixed, "sensitivity": sens_fixed, "specificity": spec_fixed})

    cal_df_out = pd.DataFrame(cal_results)
    fixed_df_out = pd.DataFrame(fixed_results)
    cal_df_out.to_csv(os.path.join(LOGS_DIR, "resnet50_500_caldriven.csv"), index=False)
    fixed_df_out.to_csv(os.path.join(LOGS_DIR, "resnet50_500_fixed05.csv"), index=False)

    print(f"\n{'='*60}\nFIXED THRESHOLD 0.5 SUMMARY\n{'='*60}")
    for metric in ["accuracy", "auc", "sensitivity", "specificity"]:
        vals = fixed_df_out[metric]
        print(f"{metric}: {vals.mean():.4f} ± {vals.std():.4f}")

    print(f"\n{'='*60}\nCALIBRATION-DRIVEN THRESHOLD SUMMARY\n{'='*60}")
    for metric in ["accuracy", "auc", "sensitivity", "specificity"]:
        vals = cal_df_out[metric]
        print(f"{metric}: {vals.mean():.4f} ± {vals.std():.4f}")


if __name__ == "__main__":
    main()