import pandas as pd
import os

PROCESSED_DIR = r"D:\Breast_Cancer_DICOM\processed"
META_DIR = r"D:\Breast_Cancer_DICOM\metadata"

# reload original mass/calc CSVs to pull subtlety scores (not carried into master_labels)
def load_csv(name):
    return pd.read_csv(os.path.join(META_DIR, name))

mass_train = load_csv("mass_case_description_train_set.csv")
mass_test = load_csv("mass_case_description_test_set.csv")
calc_train = load_csv("calc_case_description_train_set.csv")
calc_test = load_csv("calc_case_description_test_set.csv")

for df in [mass_train, mass_test, calc_train, calc_test]:
    df.rename(columns={"breast_density": "breast density"}, inplace=True, errors="ignore")

subtlety_map = pd.concat([mass_train, mass_test, calc_train, calc_test], ignore_index=True)
subtlety_map["key"] = (subtlety_map["patient_id"] + "_" +
                        subtlety_map["left or right breast"] + "_" +
                        subtlety_map["image view"] + "_" +
                        subtlety_map["abnormality id"].astype(str))

df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_labels_final.csv"))

# ---- Filter 1: drop BENIGN_WITHOUT_CALLBACK -- ambiguous label category ----
before = len(df)
df = df[df["pathology"].isin(["MALIGNANT", "BENIGN"])].reset_index(drop=True)
print(f"After dropping BENIGN_WITHOUT_CALLBACK: {len(df)} / {before} rows kept")

# ---- Filter 2: join subtlety and keep subtlety >= 3 ----
# match on patient_id + side + view (abnormality id not preserved cleanly through our pipeline,
# so we take the max subtlety per patient/side/view group as a conservative proxy)
subtlety_lookup = subtlety_map.groupby(
    ["patient_id", "left or right breast", "image view"]
)["subtlety"].max().reset_index()
subtlety_lookup.rename(columns={
    "left or right breast": "side", "image view": "view"
}, inplace=True)

df = df.merge(subtlety_lookup, on=["patient_id", "side", "view"], how="left")

before = len(df)
df_curated = df[df["subtlety"] >= 3].reset_index(drop=True)
print(f"After subtlety >= 3 filter: {len(df_curated)} / {before} rows kept")

print("\n=== Curated pathology distribution ===")
print(df_curated["pathology"].value_counts())

print("\n=== Patients remaining ===")
print(df_curated["patient_id"].nunique())

out_path = os.path.join(PROCESSED_DIR, "master_labels_curated.csv")
df_curated.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")