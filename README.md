# PVC Site-of-Origin Classification from 12-Lead ECG

**Computational Biomedical Engineering Seminar 2026 — Group E**

Classification of the **site of origin (SOO)** of premature ventricular contractions (PVCs) from 12-lead ECG signals using QRS morphology and clinical features.

---

## Project Overview

| | Description |
|---|---|
| **Task 1** | Binary classification: RVOT vs. LVOT |
| **Task 2** | 3-class classification: RVOT · RCC-LCC-Commissure · LVOT |
| **Primary dataset** | Teknon (held-out test set, N=36 patients) |
| **Training pool** | Teknon + CARTO2 + Database2 + Sims2 (N≈3009 signals) |

---

## Repository Structure

```
├── data/                          # Place data files here (see Data section)
├── Task1/
│   ├── exp_1/
│   │   └── Experiment1_QRS_morphology.ipynb   # Exp1: Sequential pipeline (binned QRS, XGBoost)
│   ├── exp_2/
│   │   └── exp2.ipynb                          # Exp2: Clinical + QRS features (Teknon only)
│   ├── exp_3/
│   │   └── exp3.ipynb                          # Exp3: 1D CNN with Grad-CAM
│   └── binned/
│       └── E3_qrs_binned_interpretable_experiment3_UPDATED_PATHS.ipynb
├── Task2/
│   ├── Exp1/
│   │   └── task2_alldatasets.py   # Exp1: Sequential pipeline, multi-dataset, 3-class
│   ├── Exp2/
│   │   └── task2_teknon.py        # Exp2: Feature families A/B/C, stacking (Teknon only)
│   └── Exp3/
│       └── CNN_task2.py           # Exp3: 2D CNN (12×200) with Grad-CAM, 3-class
├── requirements.txt
└── README.md
```

---

## Data Preprocessing

Raw data from the four source datasets (Teknon, CARTO2, Database2, Sims2) is 
processed by `preprocessing/introduction.ipynb`. This notebook performs:

- SOO label standardization (mapping ~90 raw label variants to `SOO_7class`, 
  with `SOO_chamber` used as a tiebreaker for ambiguous cases)
- Patient-level train/validation/test splitting
- Imputation (median for continuous features, mode for binary features) and 
  encoding, fit only on training data to avoid leakage
- Generation of the final `.npy`/`.pkl` files used as input for Task1 and Task2

Run this notebook first if starting from raw data:

```bash
cd preprocessing
jupyter notebook introduction.ipynb
# Run all cells in order
# Outputs: data/*.npy, data/*.pkl files described in the Data section below
```

If you already have the processed `data/` files, you can skip this step and 
go directly to Task 1 / Task 2.

---
---

## Data

The data files are **not included** in this repository due to size and privacy constraints.

Place the following files in the `data/` directory before running any script:

**Task 1 (binary: RVOT vs. LVOT):**
```
X_train_pool_binary_corrected_clean.npy       # (3009, 12, 200)
y_train_pool_binary_corrected_clean.npy
info_train_pool_binary_corrected_clean.csv
X_teknon_final_test_binary_corrected_clean.npy # (36, 12, 200)
y_teknon_final_test_binary_corrected_clean.npy
info_teknon_final_test_binary_corrected_clean.csv
full_data_7class.pkl
```

**Task 2 (3-class: RVOT · RCC-LCC · LVOT):**
```
X_train_pool_task2_3class.npy
y_train_pool_task2_3class.npy
X_teknon_final_test_task2_3class.npy
y_teknon_final_test_task2_3class.npy
full_data_7class.pkl
```

> Label convention — Task 1: `0 = LVOT`, `1 = RVOT`.  
> Label convention — Task 2: `0 = RVOT`, `1 = RCC-LCC-Commissure`, `2 = LVOT`.

---

## Results

### Task 1 — Binary Classification (RVOT vs. LVOT), Teknon test set (N=36)

| Experiment | Model | Balanced Accuracy | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Exp1** — Binned QRS, sequential pipeline | XGBoost | 0.778 | 0.778 | 0.775 | 0.818 |
| **Exp2** — Clinical + QRS probabilities (best) | Logistic Regression | **0.861** | **0.861** | **0.861** | **0.929** |
| Exp2 — Clinical only | Logistic Regression | 0.833 | 0.833 | 0.833 | 0.917 |
| **Exp3** — 1D CNN + Grad-CAM | CNN | 0.722 | 0.722 | 0.721 | 0.778 |

**Best Task 1 model:** Exp2 — Logistic Regression on Clinical + QRS OOF Probabilities (Family C2).

---

### Task 2 — 3-Class Classification (RVOT · RCC-LCC · LVOT), Teknon test set (N=35)

| Experiment | Model | Balanced Accuracy | Accuracy | Macro F1 | Macro ROC-AUC |
|---|---|---|---|---|---|
| **Exp1** — Binned QRS, all datasets | XGBoost | 0.530 | 0.571 | 0.521 | 0.680 |
| **Exp2** — Feature families A/B/C, stacking | Random Forest (Family C) | **0.547** | **0.629** | **0.526** | — |
| **Exp3** — 2D CNN + Grad-CAM | CNN | 0.471 | 0.486 | 0.449 | 0.601 |

**Best Task 2 model:** Exp2 — Random Forest on Family C (QRS OOF probabilities + clinical features).  
Per-class F1 (Exp2 best): RVOT ≈ 0.80 · RCC-LCC recall ≈ 0.20 · LVOT intermediate.

---

## Setup

```bash
# Create and activate environment
conda create -n pvc_soo python=3.10
conda activate pvc_soo

# Install dependencies
pip install -r requirements.txt
```

---

## How to Reproduce Results

### Task 1

**Experiment 1** (Jupyter notebook):
```bash
cd Task1/exp_1
jupyter notebook Experiment1_QRS_morphology.ipynb
# Run all cells in order
```

**Experiment 2** (Jupyter notebook):
```bash
cd Task1/exp_2
jupyter notebook exp2.ipynb
# Run all cells in order
```

**Experiment 3 — 1D CNN** (Jupyter notebook):
```bash
cd Task1/exp_3
jupyter notebook exp3.ipynb
# Run all cells in order
```

### Task 2

**Experiment 1** (Python script):
```bash
cd Task2/Exp1
python task2_alldatasets.py
# Outputs: task2_final_test_results.csv, task2_phaseA/B/C CSVs, figures
```

**Experiment 2** (Python script):
```bash
cd Task2/Exp2
python task2_teknon.py
# Outputs: figures (confusion, ROC, importance), error audit CSV
# Final test metrics are printed to stdout
```

**Experiment 3 — 2D CNN** (Python script):
```bash
cd Task2/Exp3
python CNN_task2.py
# Outputs: E4_final_model_weights.pth, E4_cnn_3class_test_results.csv, Grad-CAM figures
# GPU/MPS used automatically if available; falls back to CPU
```

---

## Pre-trained Model

The final CNN model weights for Task 2 Exp3 are available at:

**`Task2/Exp3/E4_final_model_weights.pth`** (included in repository, ~72 KB)

To load the model:
```python
import torch
from CNN_task2 import ResNet1D  # or the model class defined in CNN_task2.py

model = ResNet1D(n_classes=3)
model.load_state_dict(torch.load("E4_final_model_weights.pth", map_location="cpu"))
model.eval()
```

For classical ML models (Task 1 Exp2, Task 2 Exp2), re-run the scripts to retrain — training takes under 2 minutes on CPU.

---

## Dependencies

See `requirements.txt`. Core libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `matplotlib`, `shap`.

---

## References

- Doste et al. (2022). *Computers in Biology and Medicine.*
- Bocanegra-Pérez et al. (2024). *Frontiers in Cardiovascular Medicine.*
- Saglietto et al. (2024). *Frontiers in Physiology.* (1D CNN architecture reference)
