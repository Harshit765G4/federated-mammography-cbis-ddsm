import pandas as pd
import numpy as np
from scipy import stats
import os

LOGS_DIR = r"D:\Breast_Cancer_DICOM\outputs\logs"

def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / denom
    return (max(0, center - margin), min(1, center + margin))


def summarize_with_ci(csv_path, label):
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    # per-fold accuracy CI needs raw n (val set size) -- reconstruct from sens/spec + counts
    # simplest robust route: pooled accuracy across all folds using mean acc * approx n
    # NOTE: for exact CI, ideally recompute from raw predictions; here we approximate
    # using reported accuracy and typical fold size printed during training.
    mean_acc = df["accuracy"].mean()
    std_acc = df["accuracy"].std()

    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f} (fold std)")

    # per-fold Wilson CI (using n=fold val set size -- update N_PER_FOLD if known exactly)
    # Fallback: report fold-level CI using accuracy as p_hat and an assumed n.
    # Best practice: pass exact n per fold if available.
    for idx, row in df.iterrows():
        # crude n estimate note -- replace with real fold n if you have it saved
        print(f"  Fold {int(row['fold'])}: acc={row['accuracy']:.4f}, auc={row['auc']:.4f}, "
              f"sens={row['sensitivity']:.4f}, spec={row['specificity']:.4f}")

    mean_auc = df["auc"].mean()
    std_auc = df["auc"].std()
    mean_sens = df["sensitivity"].mean()
    std_sens = df["sensitivity"].std()
    mean_spec = df["specificity"].mean()
    std_spec = df["specificity"].std()

    print(f"\nPooled summary:")
    print(f"  Accuracy:    {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  AUC:         {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Sensitivity: {mean_sens:.4f} ± {std_sens:.4f}")
    print(f"  Specificity: {mean_spec:.4f} ± {std_spec:.4f}")


summarize_with_ci(os.path.join(LOGS_DIR, "resnet50_curated_v2_fixed05.csv"), "FIXED THRESHOLD 0.5 (primary)")
summarize_with_ci(os.path.join(LOGS_DIR, "resnet50_curated_v2_caldriven.csv"), "CALIBRATION-DERIVED THRESHOLD (secondary)")