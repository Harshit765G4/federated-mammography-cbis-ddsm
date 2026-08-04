import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

PROCESSED_DIR = r"D:\Breast_Cancer_DICOM\processed"
IMG_SRC_DIR = os.path.join(PROCESSED_DIR, "images")
IMG_OUT_DIR = os.path.join(PROCESSED_DIR, "images_clahe")
QA_DIR = os.path.join(PROCESSED_DIR, "qa_preview")
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(QA_DIR, exist_ok=True)

CLIP_LIMIT = 3.0
SIZE = 512

df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_labels_with_folds.csv"))

clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(8, 8))

out_paths = []
failed_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="CLAHE preprocessing"):
    src = os.path.join(IMG_SRC_DIR, row["filename"])
    try:
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("cv2.imread returned None")
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        img_clahe = clahe.apply(img)

        out_path = os.path.join(IMG_OUT_DIR, row["filename"])
        cv2.imwrite(out_path, img_clahe)
        out_paths.append(out_path)
    except Exception as e:
        out_paths.append(None)
        failed_rows.append((idx, src, str(e)))

df["clahe_image_path"] = out_paths

before = len(df)
df_clean = df[df["clahe_image_path"].notna()].reset_index(drop=True)
print(f"\nRows before: {before}, after dropping failures: {len(df_clean)}")
print(f"Failed rows: {len(failed_rows)}")
if failed_rows:
    print("Sample failures:", failed_rows[:5])

final_path = os.path.join(PROCESSED_DIR, "master_labels_final.csv")
df_clean.to_csv(final_path, index=False)
print(f"Saved: {final_path}")

print("\n=== VERIFICATION ===")
missing_files = df_clean["clahe_image_path"].apply(lambda p: not os.path.isfile(p)).sum()
print("Output files missing on disk:", missing_files)

sample = df_clean.sample(min(50, len(df_clean)), random_state=42)
bad_shape = 0
value_ranges = []
for _, row in sample.iterrows():
    img = cv2.imread(row["clahe_image_path"], cv2.IMREAD_GRAYSCALE)
    if img.shape != (SIZE, SIZE):
        bad_shape += 1
    value_ranges.append((img.min(), img.max()))

print(f"Sample size checked: {len(sample)}")
print("Wrong shape count:", bad_shape, "(should be 0)")
mins = [v[0] for v in value_ranges]
maxs = [v[1] for v in value_ranges]
print(f"Pixel value range: min={min(mins)}, max={max(maxs)}")

# visual QA grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

qa_sample = df_clean.sample(6, random_state=1)
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for i, (_, row) in enumerate(qa_sample.iterrows()):
    before_img = cv2.imread(os.path.join(IMG_SRC_DIR, row["filename"]), cv2.IMREAD_GRAYSCALE)
    after_img = cv2.imread(row["clahe_image_path"], cv2.IMREAD_GRAYSCALE)
    axes[0, i].imshow(before_img, cmap="gray")
    axes[0, i].set_title(f"Before\n{row['pathology'][:8]}", fontsize=8)
    axes[0, i].axis("off")
    axes[1, i].imshow(after_img, cmap="gray")
    axes[1, i].set_title("After CLAHE", fontsize=8)
    axes[1, i].axis("off")

plt.tight_layout()
fig_path = os.path.join(QA_DIR, "before_after_grid.png")
plt.savefig(fig_path, dpi=120)
print(f"\nSaved visual QA grid: {fig_path}")