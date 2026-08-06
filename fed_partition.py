import pandas as pd
import numpy as np
import os

PROCESSED_DIR = r"D:\Breast_Cancer_DICOM\processed"

def partition_iid(df, n_clients=3, seed=42):
    """Split patients roughly evenly and randomly across clients."""
    patients = df["patient_id"].drop_duplicates().sample(frac=1, random_state=seed).tolist()
    splits = np.array_split(patients, n_clients)
    client_dfs = []
    for i, pat_list in enumerate(splits):
        client_df = df[df["patient_id"].isin(pat_list)].reset_index(drop=True)
        client_dfs.append(client_df)
    return client_dfs


def partition_noniid_dirichlet(df, n_clients=3, alpha=0.5, seed=42):
    """
    Dirichlet-based non-IID partition by pathology label, patient-wise.
    Each patient assigned fully to one client (no patient split across clients).
    """
    rng = np.random.default_rng(seed)

    # assign each patient a single dominant label (majority pathology among their images)
    patient_label = df.groupby("patient_id")["pathology"].agg(
        lambda x: x.value_counts().idxmax()
    ).reset_index()
    patient_label.columns = ["patient_id", "dominant_label"]

    client_patient_ids = [[] for _ in range(n_clients)]

    for label in patient_label["dominant_label"].unique():
        label_patients = patient_label[patient_label["dominant_label"] == label]["patient_id"].tolist()
        rng.shuffle(label_patients)

        proportions = rng.dirichlet(alpha=[alpha] * n_clients)
        proportions = (np.cumsum(proportions) * len(label_patients)).astype(int)[:-1]
        splits = np.split(label_patients, proportions)

        for i, pat_list in enumerate(splits):
            client_patient_ids[i].extend(pat_list)

    client_dfs = []
    for pat_list in client_patient_ids:
        client_df = df[df["patient_id"].isin(pat_list)].reset_index(drop=True)
        client_dfs.append(client_df)
    return client_dfs


def print_partition_summary(client_dfs, label):
    print(f"\n=== {label} ===")
    for i, cdf in enumerate(client_dfs):
        counts = cdf["pathology"].value_counts()
        n_patients = cdf["patient_id"].nunique()
        print(f"Client {i}: {len(cdf)} images, {n_patients} patients, "
              f"MALIGNANT={counts.get('MALIGNANT', 0)}, BENIGN={counts.get('BENIGN', 0)}")


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_labels_curated.csv"))

    # global held-out test set: use fold 0 as test, remaining folds 1-4 for federated clients
    test_df = df[df["fold"] == 0].reset_index(drop=True)
    fed_df = df[df["fold"] != 0].reset_index(drop=True)

    print(f"Global test set: {len(test_df)} images, {test_df['patient_id'].nunique()} patients")
    print(f"Federated pool: {len(fed_df)} images, {fed_df['patient_id'].nunique()} patients")

    iid_clients = partition_iid(fed_df, n_clients=3)
    print_partition_summary(iid_clients, "IID PARTITION")

    noniid_clients = partition_noniid_dirichlet(fed_df, n_clients=3, alpha=0.5)
    print_partition_summary(noniid_clients, "NON-IID PARTITION (Dirichlet alpha=0.5)")

    # save partitions to disk for reuse across FedAvg/FedProx/SCAFFOLD scripts
    test_df.to_csv(os.path.join(PROCESSED_DIR, "fed_test_set.csv"), index=False)
    for i, cdf in enumerate(iid_clients):
        cdf.to_csv(os.path.join(PROCESSED_DIR, f"fed_iid_client{i}.csv"), index=False)
    for i, cdf in enumerate(noniid_clients):
        cdf.to_csv(os.path.join(PROCESSED_DIR, f"fed_noniid_client{i}.csv"), index=False)

    print("\nSaved all partitions to processed/")