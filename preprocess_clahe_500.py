import pandas as pd
import cv2
import os
from tqdm import tqdm

PROCESSED_DIR = r"D:\Breast_Cancer_DICOM\processed"
IMG_SRC_DIR = os.path.join(PROCESSED_DIR, "images")
IMG_OUT_DIR = os.path.join(PROCESSED_DIR, "images_clahe_500")
os.makedirs(IMG_OUT_DIR, exist_ok=True)

CLIP_LIMIT = 3.0
SIZE = 512

df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_labels_500.csv"))
clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(8, 8))

out_paths = []
failed = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="CLAHE 500-subset"):
    src = os.path.join(IMG_SRC_DIR, row["filename"])
    try:
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("read failed")
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        img_clahe = clahe.apply(img)
        out_path = os.path.join(IMG_OUT_DIR, row["filename"])
        cv2.imwrite(out_path, img_clahe)
        out_paths.append(out_path)
    except Exception as e:
        out_paths.append(None)
        failed.append((idx, src, str(e)))

df["clahe_image_path"] = out_paths
df_clean = df[df["clahe_image_path"].notna()].reset_index(drop=True)

print(f"\nProcessed: {len(df_clean)} / {len(df)}, Failed: {len(failed)}")
if failed:
    print("Sample failures:", failed[:5])

out_csv = os.path.join(PROCESSED_DIR, "master_labels_500_final.csv")
df_clean.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")