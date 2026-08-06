import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import os
import time
import copy
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

PROCESSED_DIR = r"D:\Breast_Cancer_DICOM\processed"
LOGS_DIR = r"D:\Breast_Cancer_DICOM\outputs\logs"
WEIGHTS_DIR = r"D:\Breast_Cancer_DICOM\outputs\weights_fedavg"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
BATCH_SIZE = 8
LOCAL_EPOCHS = 2
N_ROUNDS = 15
N_CLIENTS = 3
LR = 1e-4
DROPOUT_P = 0.5


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


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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


def get_trainable_state(model):
    """Trainable params + BN buffers from unfrozen layers (layer3, layer4)."""
    state = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.detach().cpu().clone()
    # include BN running stats for BN layers inside layer3/layer4 (unfrozen)
    for name, buf in model.named_buffers():
        if name.startswith("layer3") or name.startswith("layer4"):
            state[name] = buf.detach().cpu().clone()
    return state


from collections import OrderedDict

def set_bn_eval_for_frozen(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def local_train(global_state, client_df, local_epochs=LOCAL_EPOCHS):
    """Train one client starting from global_state, return updated trainable state + n_samples."""
    model = build_model()
    full_state = model.state_dict()
    full_state.update(global_state)
    model.load_state_dict(full_state)

    train_ds = MammoDataset(client_df, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    class_counts = client_df["pathology"].apply(lambda p: 1 if p == "MALIGNANT" else 0).value_counts()
    n_benign = class_counts.get(0, 1)
    n_malignant = class_counts.get(1, 1)
    raw_ratio = n_benign / max(n_malignant, 1)
    capped_ratio = min(max(raw_ratio, 0.33), 3.0)  # cap between 1:3 and 3:1
    weight = torch.tensor([1.0, capped_ratio], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-5)

    model.train()
    set_bn_eval_for_frozen(model)

    start = time.time()
    for epoch in range(local_epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
    train_time = time.time() - start

    return get_trainable_state(model), len(client_df), train_time


def fedavg_aggregate(client_states, client_sizes):
    """Weighted average of client states by number of local samples."""
    total_samples = sum(client_sizes)
    avg_state = OrderedDict()
    for key in client_states[0].keys():
        avg_state[key] = sum(
            client_states[i][key] * (client_sizes[i] / total_samples)
            for i in range(len(client_states))
        )
    return avg_state


def evaluate_global_model(global_state, test_df):
    model = build_model()
    full_state = model.state_dict()
    full_state.update(global_state)
    model.load_state_dict(full_state)

    test_ds = MammoDataset(test_df, transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            probs = torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    return acc, auc, sens, spec, model


def run_fedavg(setting_name, client_dfs, test_df):
    print(f"\n{'='*60}\nFEDAVG - {setting_name}\n{'='*60}")

    global_model = build_model()
    global_state = get_trainable_state(global_model)
    del global_model

    round_metrics = []

    for rnd in range(1, N_ROUNDS + 1):
        client_states = []
        client_sizes = []
        round_train_time = 0

        for cid, client_df in enumerate(client_dfs):
            state, n_samples, t = local_train(global_state, client_df)
            client_states.append(state)
            client_sizes.append(n_samples)
            round_train_time += t

        global_state = fedavg_aggregate(client_states, client_sizes)

        acc, auc, sens, spec, eval_model = evaluate_global_model(global_state, test_df)
        round_metrics.append({
            "round": rnd, "accuracy": acc, "auc": auc,
            "sensitivity": sens, "specificity": spec,
            "train_time_sec": round_train_time
        })
        print(f"Round {rnd}: acc={acc:.4f}, auc={auc:.4f}, sens={sens:.4f}, spec={spec:.4f}, "
              f"train_time={round_train_time:.1f}s")

    torch.save(eval_model.state_dict(), os.path.join(WEIGHTS_DIR, f"fedavg_{setting_name}_final.pt"))

    results_df = pd.DataFrame(round_metrics)
    results_df.to_csv(os.path.join(LOGS_DIR, f"fedavg_{setting_name}_rounds.csv"), index=False)

    final = round_metrics[-1]
    print(f"\nFINAL ({setting_name}): acc={final['accuracy']:.4f}, auc={final['auc']:.4f}, "
          f"sens={final['sensitivity']:.4f}, spec={final['specificity']:.4f}")
    return final


if __name__ == "__main__":
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "fed_test_set.csv"))

    iid_clients = [pd.read_csv(os.path.join(PROCESSED_DIR, f"fed_iid_client{i}.csv")) for i in range(N_CLIENTS)]
    noniid_clients = [pd.read_csv(os.path.join(PROCESSED_DIR, f"fed_noniid_client{i}.csv")) for i in range(N_CLIENTS)]

    iid_result = run_fedavg("iid", iid_clients, test_df)
    noniid_result = run_fedavg("noniid", noniid_clients, test_df)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"FedAvg IID:     acc={iid_result['accuracy']:.4f}, auc={iid_result['auc']:.4f}")
    print(f"FedAvg non-IID: acc={noniid_result['accuracy']:.4f}, auc={noniid_result['auc']:.4f}")
    print(f"\nCompare to your centralized baseline: acc≈0.69, auc≈0.79")