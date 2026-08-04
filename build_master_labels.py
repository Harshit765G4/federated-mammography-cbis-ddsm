import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold

PROCESSED_DIR = r"D:\Breast_Cancer_DICOM\processed"

df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_labels.csv"))

X = df["filename"]
y = df["pathology"]
groups = df["patient_id"]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    df.loc[val_idx, "fold"] = fold

out_path = os.path.join(PROCESSED_DIR, "master_labels_with_folds.csv")
df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print("Final row count:", len(df))

print("\n=== Pathology distribution ===")
print(df["pathology"].value_counts())

print("\n=== Images per patient (describe) ===")
print(df.groupby("patient_id").size().describe())

print("\n=== Fold distribution ===")
print(df["fold"].value_counts().sort_index())

print("\n=== Leakage check: any patient in >1 fold? ===")
leak = df.groupby("patient_id")["fold"].nunique()
print("Patients spanning multiple folds:", (leak > 1).sum(), "(should be 0)")