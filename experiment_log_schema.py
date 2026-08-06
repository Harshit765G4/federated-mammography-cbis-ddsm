"""
Central experiment log schema. Every FedAvg/FedProx/SCAFFOLD/DP-SGD run appends
one row here. Run this once to create the empty log file.
"""
import pandas as pd
import os

LOGS_DIR = r"D:\Breast_Cancer_DICOM\outputs\logs"
LOG_PATH = os.path.join(LOGS_DIR, "master_experiment_log.csv")

COLUMNS = [
    "run_id", "algorithm", "iid_setting", "epsilon", "seed",
    "n_clients", "local_epochs", "n_rounds",
    "accuracy", "auc", "sensitivity", "specificity",
    "n_test", "accuracy_ci_low", "accuracy_ci_high",
    "training_time_sec_per_round", "communication_time_sec_per_round",
    "cumulative_bandwidth_gb", "notes"
]

if not os.path.exists(LOG_PATH):
    pd.DataFrame(columns=COLUMNS).to_csv(LOG_PATH, index=False)
    print(f"Created: {LOG_PATH}")
else:
    print(f"Already exists: {LOG_PATH}")

def append_run(row_dict):
    """Call this at the end of every training run to log a result."""
    df = pd.read_csv(LOG_PATH)
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"Logged run: {row_dict.get('run_id', 'unknown')}")