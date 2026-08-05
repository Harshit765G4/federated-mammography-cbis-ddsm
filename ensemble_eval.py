import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import os

LOGS_DIR = r"D:\Breast_Cancer_DICOM\outputs\logs"
N_FOLDS = 5

def optimal_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    return thresholds[np.argmax(j_scores)]

for weight_resnet in [0.5, 0.6, 0.7, 0.8]:
    fold_metrics = []
    for fold in range(N_FOLDS):
        r = pd.read_csv(os.path.join(LOGS_DIR, f"resnet_fold{fold}_val_probs.csv"))
        d = pd.read_csv(os.path.join(LOGS_DIR, f"densenet_fold{fold}_val_probs.csv"))

        merged = r.merge(d[["filename", "densenet_prob"]], on="filename")
        merged["ensemble_prob"] = weight_resnet * merged["resnet_prob"] + (1 - weight_resnet) * merged["densenet_prob"]
        labels = (merged["pathology"] == "MALIGNANT").astype(int)

        auc = roc_auc_score(labels, merged["ensemble_prob"])
        thresh = optimal_threshold(labels, merged["ensemble_prob"])
        preds = (merged["ensemble_prob"] > thresh).astype(int)
        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)

        fold_metrics.append({"acc": acc, "auc": auc, "sens": sens, "spec": spec})

    fm = pd.DataFrame(fold_metrics)
    print(f"\nResNet weight={weight_resnet}: acc={fm['acc'].mean():.4f}±{fm['acc'].std():.4f}, "
          f"auc={fm['auc'].mean():.4f}±{fm['auc'].std():.4f}, "
          f"sens={fm['sens'].mean():.4f}, spec={fm['spec'].mean():.4f}")