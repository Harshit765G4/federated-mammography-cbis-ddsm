import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import os
import time
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

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


def get_trainable_params(model):
    return [val.detach().cpu().numpy() for name, val in model.named_parameters() if val.requires_grad]


def set_trainable_params(model, params):
    trainable_names = [name for name, val in model.named_parameters() if val.requires_grad]
    params_dict = zip(trainable_names, params)
    state_dict = model.state_dict()
    for name, param in params_dict:
        state_dict[name] = torch.tensor(param)
    model.load_state_dict(state_dict, strict=False)


def set_bn_eval_for_frozen(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


class MammoClient(fl.client.NumPyClient):
    def __init__(self, client_df):
        self.model = build_model()
        train_ds = MammoDataset(client_df, transform=train_transform)
        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        class_counts = client_df["pathology"].apply(lambda p: 1 if p == "MALIGNANT" else 0).value_counts()
        n_benign = class_counts.get(0, 1)
        n_malignant = class_counts.get(1, 1)
        weight = torch.tensor([1.0, n_benign / max(n_malignant, 1)], dtype=torch.float32).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def get_parameters(self, config):
        return get_trainable_params(self.model)

    def fit(self, parameters, config):
        set_trainable_params(self.model, parameters)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=LR, weight_decay=1e-5)

        self.model.train()
        set_bn_eval_for_frozen(self.model)

        start = time.time()
        for epoch in range(LOCAL_EPOCHS):
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = self.criterion(self.model(imgs), labels)
                loss.backward()
                optimizer.step()
        train_time = time.time() - start

        return get_trainable_params(self.model), len(self.train_loader.dataset), {"train_time": train_time}

    def evaluate(self, parameters, config):
        set_trainable_params(self.model, parameters)
        self.model.eval()
        return 0.0, 1, {}


def make_client_fn(client_dfs):
    def client_fn(context):
        cid = int(context.node_config["partition-id"])
        return MammoClient(client_dfs[cid]).to_client()
    return client_fn


def evaluate_global_model(model, test_df):
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
    return acc, auc, sens, spec


def run_fedavg(setting_name, client_dfs, test_df):
    print(f"\n{'='*60}\nFEDAVG - {setting_name}\n{'='*60}")

    global_model = build_model()
    init_params = ndarrays_to_parameters(get_trainable_params(global_model))

    round_metrics = []

    class LoggingFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
            if aggregated_params is not None:
                ndarrays = parameters_to_ndarrays(aggregated_params)
                set_trainable_params(global_model, ndarrays)
                acc, auc, sens, spec = evaluate_global_model(global_model, test_df)
                round_metrics.append({
                    "round": server_round, "accuracy": acc, "auc": auc,
                    "sensitivity": sens, "specificity": spec
                })
                print(f"Round {server_round}: acc={acc:.4f}, auc={auc:.4f}, sens={sens:.4f}, spec={spec:.4f}")
            return aggregated_params, metrics

    strategy = LoggingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=N_CLIENTS,
        min_available_clients=N_CLIENTS,
        initial_parameters=init_params,
    )

    fl.simulation.start_simulation(
        client_fn=make_client_fn(client_dfs),
        num_clients=N_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 1.0 if DEVICE.type == "cuda" else 0.0},
    )

    torch.save(global_model.state_dict(), os.path.join(WEIGHTS_DIR, f"fedavg_{setting_name}_final.pt"))

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