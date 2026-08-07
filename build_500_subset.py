import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold

PROCESSED_DIR = r"D:\Breast_Cancer_DICOM\processed"
META_DIR = r"D:\Breast_Cancer_DICOM\metadata"

def load_csv(name):
    return pd.read_csv(os.path.join(META_DIR, name))

mass_train = load_csv("mass_case_description_train_set.csv")
mass_test = load_csv("mass_case_description_test_set.csv")
calc_train = load_csv("calc_case_description_train_set.csv")
calc_test = load_csv("calc_case_description_test_set.csv")

for df in [mass_train, mass_test, calc_train, calc_test]:
    df.rename(columns={"breast_density": "breast density"}, inplace=True, errors="ignore")

subtlety_map = pd.concat([mass_train, mass_test, calc_train, calc_test], ignore_index=True)
subtlety_lookup = subtlety_map.groupby(
    ["patient_id", "left or right breast", "image view"]
)["subtlety"].max().reset_index()
subtlety_lookup.rename(columns={"left or right breast": "side", "image view": "view"}, inplace=True)

df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_labels_final.csv"))
df = df[df["pathology"].isin(["MALIGNANT", "BENIGN"])].reset_index(drop=True)
df = df.merge(subtlety_lookup, on=["patient_id", "side", "view"], how="left")

# ---- Selection: highest subtlety first, patient-wise (one image per patient to maximize diversity) ----
# take one representative image per patient (their highest-subtlety image) to avoid
# over-representing patients with many images, then rank patients by subtlety within class
df_sorted = df.sort_values("subtlety", ascending=False)
one_per_patient = df_sorted.drop_duplicates(subset="patient_id", keep="first").reset_index(drop=True)

malignant_pool = one_per_patient[one_per_patient["pathology"] == "MALIGNANT"].sort_values("subtlety", ascending=False)
benign_pool = one_per_patient[one_per_patient["pathology"] == "BENIGN"].sort_values("subtlety", ascending=False)

print(f"Available MALIGNANT (1/patient, sorted by subtlety): {len(malignant_pool)}")
print(f"Available BENIGN (1/patient, sorted by subtlety): {len(benign_pool)}")
print(f"\nMALIGNANT subtlety distribution:\n{malignant_pool['subtlety'].value_counts().sort_index()}")
print(f"\nBENIGN subtlety distribution:\n{benign_pool['subtlety'].value_counts().sort_index()}")

N_PER_CLASS = 250

selected_malignant = malignant_pool.head(N_PER_CLASS)
selected_benign = benign_pool.head(N_PER_CLASS)

subset_500 = pd.concat([selected_malignant, selected_benign], ignore_index=True)

print(f"\n=== Final 500-image subset ===")
print(subset_500["pathology"].value_counts())
print(f"Patients: {subset_500['patient_id'].nunique()} (should be 500, one image per patient)")
print(f"\nSubtlety distribution in final subset:")
print(subset_500.groupby("pathology")["subtlety"].describe())

# ---- Rebuild folds fresh on this subset (critical -- do not reuse old fold assignments) ----
X = subset_500["filename"]
y = subset_500["pathology"]
groups = subset_500["patient_id"]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
subset_500["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    subset_500.loc[val_idx, "fold"] = fold

print(f"\n=== Fold distribution ===")
print(subset_500["fold"].value_counts().sort_index())
print(f"\n=== Fold x Pathology ===")
print(subset_500.groupby("fold")["pathology"].value_counts().unstack())

leak = subset_500.groupby("patient_id")["fold"].nunique()
print(f"\nPatients spanning multiple folds: {(leak > 1).sum()} (must be 0)")

out_path = os.path.join(PROCESSED_DIR, "master_labels_500.csv")
subset_500.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")