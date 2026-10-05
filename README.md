# 🩺 Federated Mammography Classification on CBIS-DDSM

A research-oriented deep-learning and federated-learning pipeline for mammography classification using the CBIS-DDSM dataset.

The repository combines DICOM/metadata extraction, CLAHE preprocessing, patient-aware dataset curation, ResNet-50 and DenseNet-121 baselines, probability-level ensembling, IID/non-IID client partitioning, and FedAvg experiments.

> **Research / medical-use note:** This is an experimental research codebase, not a clinical diagnostic system.

## 🎯 Research Objective

The current workflow is:

```text
CBIS-DDSM Metadata + DICOM
            │
            ▼
    Master Label Construction
            │
            ▼
       CLAHE Processing
            │
            ▼
     Curated / Full Dataset
            │
            ▼
 Patient-wise 5-fold evaluation
            │
      ┌─────┴─────┐
      ▼           ▼
ResNet-50     DenseNet-121
      │           │
      └─────┬─────┘
            ▼
     Probability Ensemble
            │
            ▼
   Federated partitions
      ┌─────┴─────┐
      ▼           ▼
     IID       non-IID
      │           │
      └─────┬─────┘
            ▼
          FedAvg
```

## 🗂️ Dataset

Metadata files are included under `metadata/`:

- `calc_case_description_train_set.csv`
- `calc_case_description_test_set.csv`
- `mass_case_description_train_set.csv`
- `mass_case_description_test_set.csv`

The raw CBIS-DDSM DICOM/image corpus is not committed to the repository. Several scripts expect local processed directories such as `D:\Breast_Cancer_DICOM\processed`.

## 🧹 Data Construction

### Curated subset

`build_curated_subset.py`:

- removes `BENIGN_WITHOUT_CALLBACK`
- joins subtlety information
- keeps cases with **subtlety >= 3**

### 500-image subset

`build_500_subset.py`:

- selects one representative image per patient
- prioritizes highest subtlety
- selects **250 malignant + 250 benign**
- rebuilds patient-wise stratified folds with seed 42

## 🩻 Image Preprocessing

The preprocessing pipeline includes DICOM extraction, grayscale conversion, resizing, normalization, and CLAHE.

Typical CNN input:

```text
512 x 512 x 3
```

Training scripts use ImageNet normalization and small geometric augmentations.

## 🤖 Centralized Models

### ResNet-50

The ResNet experiments use ImageNet-pretrained ResNet-50 with a 2-class replacement head and fine-tuning of deeper layers. Training includes class-weighted cross-entropy, Adam optimization, learning-rate scheduling, and early stopping.

Relevant scripts include:

```text
train_resnet50.py
train_resnet50_5fold.py
train_resnet50_500.py
train_resnet50_curated.py
train_resnet50_curated_v2.py
```

### DenseNet-121

`train_densenet121_curated.py` provides a second architecture with:

- ImageNet-pretrained DenseNet-121
- deep-block fine-tuning
- class weighting
- cosine learning-rate scheduling
- early stopping
- horizontal-flip test-time augmentation
- ROC-derived threshold selection

## 👤 Patient-Level Evaluation

Patient leakage is explicitly controlled.

For federated experiments:

- fold 0 is the global held-out test set
- folds 1-4 form the federated training pool
- all images from a patient remain on the same client

This is important because a patient may contribute multiple mammograms.

## 🔗 Probability-Level Ensemble

`generate_resnet_probs.py` exports fold-level ResNet probabilities, while DenseNet training exports its own validation probabilities.

`ensemble_eval.py` combines them as:

```text
ensemble = w * ResNet_probability
         + (1-w) * DenseNet_probability
```

Tested ResNet weights:

```text
0.5, 0.6, 0.7, 0.8
```

The evaluator reports AUC, threshold, accuracy, sensitivity, and specificity.

## 🌐 Federated Partitioning

Implemented in `fed_partition.py`.

### IID

Patients are shuffled and divided approximately evenly across **3 clients**.

### Non-IID

A patient-wise Dirichlet partition is used with:

```text
clients = 3
alpha = 0.5
seed = 42
```

Patients are assigned by their dominant pathology label and are never split across clients.

## 🔄 FedAvg

The repository contains both a manual PyTorch implementation and a Flower simulation.

### Manual FedAvg

`fedavg_manual.py` uses:

| Setting | Value |
|---|---:|
| Clients | 3 |
| Local epochs | 2 |
| Rounds | 15 |
| Batch size | 8 |
| Learning rate | 1e-4 |
| Image size | 512 |

The loop performs local client training, sample-count-weighted aggregation, and global test evaluation.

### Flower FedAvg

`fedavg_simulation.py` uses Flower's `NumPyClient` and `FedAvg` strategy for a framework-based simulation.

## 📈 Stored FedAvg Results

These are **committed results from the repository**, not freshly rerun during README generation.

### IID — final round

From `outputs/logs/fedavg_iid_rounds.csv`:

| Metric | Round 15 |
|---|---:|
| Accuracy | **0.7483** |
| AUC | **0.8229** |
| Sensitivity | 0.6351 |
| Specificity | 0.8630 |

The best stored IID AUC appears earlier, at about **0.8335** in round 2.

### Non-IID — final round

From `outputs/logs/fedavg_noniid_rounds.csv`:

| Metric | Round 15 |
|---|---:|
| Accuracy | **0.6440** |
| AUC | **0.7202** |
| Sensitivity | 0.4595 |
| Specificity | 0.8311 |

The stored non-IID run shows a clear degradation relative to the IID setting.

## 🧪 Stored ResNet-50 Results

### Full-dataset 5-fold

`outputs/logs/resnet50_5fold_results.csv` contains:

| Fold | Accuracy | AUC |
|---:|---:|---:|
| 0 | 0.7731 | 0.8639 |
| 1 | 0.7003 | 0.8338 |
| 2 | 0.7857 | 0.8727 |
| 3 | 0.7602 | 0.8422 |
| 4 | 0.6985 | 0.8280 |

### Curated v2 — fixed 0.5 threshold

`outputs/logs/resnet50_curated_v2_fixed05.csv` stores the no-leakage fixed-threshold evaluation.

Mean values recorded in the training output are approximately:

| Metric | Mean ± Std |
|---|---:|
| Accuracy | **0.6897 ± 0.0712** |
| AUC | **0.7936 ± 0.0571** |
| Sensitivity | **0.5959 ± 0.2185** |
| Specificity | **0.7809 ± 0.1637** |

A calibration-derived threshold version is also stored separately in `resnet50_curated_v2_caldriven.csv`.

## 🧪 DenseNet-121 Results

`outputs/logs/densenet121_curated_5fold_results.csv` contains five stored folds with AUC values ranging from approximately **0.765 to 0.801**.

## 📋 Reproducibility Logs

Important machine-readable outputs include:

```text
outputs/logs/
├── densenet121_curated_5fold_results.csv
├── densenet_fold0_val_probs.csv
├── densenet_fold1_val_probs.csv
├── densenet_fold2_val_probs.csv
├── densenet_fold3_val_probs.csv
├── densenet_fold4_val_probs.csv
├── fedavg_iid_rounds.csv
├── fedavg_noniid_rounds.csv
├── master_experiment_log.csv
├── resnet50_5fold_results.csv
├── resnet50_curated_5fold_results.csv
├── resnet50_curated_v2_caldriven.csv
├── resnet50_curated_v2_fixed05.csv
└── resnet50_training_history.csv
```

The repository also includes experiment-log schema and Wilson confidence-interval utilities.

## 📁 Repository Structure

```text
federated-mammography-cbis-ddsm/
├── metadata/
├── outputs/
│   └── logs/
├── build_master_labels.py
├── build_curated_subset.py
├── build_500_subset.py
├── extract_dicom.py
├── preprocess_clahe.py
├── preprocess_clahe_500.py
├── train_resnet50.py
├── train_resnet50_5fold.py
├── train_resnet50_500.py
├── train_resnet50_curated.py
├── train_resnet50_curated_v2.py
├── train_densenet121_curated.py
├── fed_partition.py
├── fedavg_manual.py
├── fedavg_simulation.py
├── generate_resnet_probs.py
├── ensemble_eval.py
├── experiment_log_schema.py
├── wilson_ci.py
├── trained_output_full_resnet50_dataset.txt
├── trained_output_resnet50_curated_dataset.txt
├── outputnew phase.txt
└── v2.txt
```

## 🚀 Typical Workflow

Adjust the hard-coded local paths before running.

### Prepare data

```bash
python extract_dicom.py
python preprocess_clahe.py
python build_master_labels.py
python build_curated_subset.py
python build_500_subset.py
```

### Train baselines

```bash
python train_resnet50_5fold.py
python train_densenet121_curated.py
```

### Generate ensemble probabilities

```bash
python generate_resnet_probs.py
python ensemble_eval.py
```

### Build federated clients

```bash
python fed_partition.py
```

### Run FedAvg

```bash
python fedavg_manual.py
python fedavg_simulation.py
```

## ⚠️ Limitations

- Raw CBIS-DDSM images are not included.
- Several scripts use hard-coded Windows paths.
- The repository contains multiple research iterations and dataset variants.
- Thresholding protocols differ between some experiments.
- FedAvg is implemented, but stronger FL variants such as FedProx and SCAFFOLD are not implemented here.
- Federated learning alone does not guarantee privacy; update leakage and inference risks still require dedicated defenses.
- The Flower setup is a simulation-style environment, not a deployed multi-hospital federation.
- Medical deployment would require external validation, calibration, governance, regulatory review, and clinical oversight.

## 🔬 Recommended Extensions

- Add FedProx and SCAFFOLD for non-IID robustness.
- Add differential privacy with explicit `epsilon`/`delta` accounting.
- Add secure aggregation.
- Test client dropout and partial participation.
- Add multiple random seeds and confidence intervals.
- Track communication cost and bandwidth per round.
- Compare centralized, IID-FL, non-IID-FL, and privacy-preserving FL on matched splits.
- Add PR-AUC and calibration curves.
- Replace hard-coded paths with CLI/configuration options.
- Consolidate repeated training code into reusable modules.

## 🩻 Medical Research Disclaimer

This software is intended for research and educational experimentation only. It is not a medical device and must not be used to make patient-care or diagnostic decisions.

## 👤 Author

**Harshit Garg**

GitHub: [Harshit765G4](https://github.com/Harshit765G4)

---

**Federated Mammography CBIS-DDSM** — patient-aware mammography classification, model comparison, and federated-learning experiments.
