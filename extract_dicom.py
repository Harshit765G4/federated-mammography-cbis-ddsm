import os
import re
import glob
import pydicom
import numpy as np
import cv2
import pandas as pd

RAW_ROOT = r"D:\cbis_ddsm"
META_DIR = r"D:\Breast_Cancer_DICOM\metadata"
OUT_DIR = r"D:\Breast_Cancer_DICOM\processed\images"
os.makedirs(OUT_DIR, exist_ok=True)

def load_csv(name):
    return pd.read_csv(os.path.join(META_DIR, name))

mass_train = load_csv("mass_case_description_train_set.csv")
mass_test  = load_csv("mass_case_description_test_set.csv")
calc_train = load_csv("calc_case_description_train_set.csv")
calc_test  = load_csv("calc_case_description_test_set.csv")

for df in [mass_train, mass_test, calc_train, calc_test]:
    df.rename(columns={"breast_density": "breast density"}, inplace=True, errors="ignore")

for df, is_train, abn_type in [
    (mass_train, True, "mass"), (mass_test, False, "mass"),
    (calc_train, True, "calc"), (calc_test, False, "calc"),
]:
    df["source_split"] = "train" if is_train else "test"
    df["abnormality_type"] = abn_type

meta = pd.concat([mass_train, mass_test, calc_train, calc_test], ignore_index=True)

def find_full_mammogram_folder(row):
    patient_id = row["patient_id"]
    side = row["left or right breast"]
    view = row["image view"]
    abn = "Mass" if row["abnormality_type"] == "mass" else "Calc"
    split_label = "Training" if row["source_split"] == "train" else "Test"
    patient_num = patient_id.replace("P_", "")

    candidates = [
        f"{abn}-{split_label}_P_{patient_num}_{side}_{view}",           # e.g. Calc-Test_P_00141_LEFT_CC (dominant pattern)
        f"{patient_id}_{side}_{view}.dcm",                              # e.g. P_00038_LEFT_CC.dcm (exception pattern)
    ]

    for name in candidates:
        full_path = os.path.join(RAW_ROOT, name)
        if os.path.isdir(full_path):
            return full_path
    return None

def find_dcm_in_folder(folder_path):
    """Descend through nested UID subfolders to find the single .dcm file."""
    matches = glob.glob(os.path.join(folder_path, "**", "*.dcm"), recursive=True)
    return matches[0] if matches else None

records = []
missing = []

for idx, row in meta.iterrows():
    folder = find_full_mammogram_folder(row)
    if folder is None:
        missing.append((idx, row["patient_id"], row["left or right breast"], row["image view"]))
        continue

    dcm_path = find_dcm_in_folder(folder)
    if dcm_path is None:
        missing.append((idx, row["patient_id"], "NO_DCM_FOUND", folder))
        continue

    try:
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(np.float32)
        # normalize 16-bit range to 0-255 for PNG storage, preserving relative contrast
        img = (img - img.min()) / (img.max() - img.min() + 1e-7) * 255.0
        img = img.astype(np.uint8)
    except Exception as e:
        missing.append((idx, row["patient_id"], "READ_ERROR", str(e)))
        continue

    fname = f"{row['patient_id']}_{row['left or right breast']}_{row['image view']}_{row['abnormality_type']}_{idx}.png"
    cv2.imwrite(os.path.join(OUT_DIR, fname), img)

    records.append({
        "filename": fname,
        "patient_id": row["patient_id"],
        "pathology": row["pathology"],
        "abnormality_type": row["abnormality_type"],
        "view": row["image view"],
        "side": row["left or right breast"],
        "split": row["source_split"],
        "source_dcm_path": dcm_path,
    })

    if idx % 200 == 0:
        print(f"Processed {idx}/{len(meta)}...")

master_df = pd.DataFrame(records)
master_df.to_csv(r"D:\Breast_Cancer_DICOM\processed\master_labels.csv", index=False)

print(f"\nConverted: {len(master_df)} / {len(meta)}")
print(f"Missing/failed: {len(missing)}")
if missing:
    print("Sample missing entries:")
    for m in missing[:10]:
        print(" ", m)