#!/usr/bin/env python3
"""
E4 — QRS CNN + Grad-CAM  ·  3-class Site-of-Origin classification
===================================================================
Extends the Saglietto-style binary CNN (E3) to a 3-macroclass problem:

    Class 0 → RVOT   (Right Ventricular Outflow Tract: Septum & Free Wall)
    Class 1 → RCC-LCC (Right/Left Coronary Cusp Commissure)
    Class 2 → LVOT   (Left Ventricular Outflow Tract: Summit & Subvalvular)

Key protocol differences from the binary notebook
──────────────────────────────────────────────────
• Framework        : PyTorch (not TensorFlow / Keras)
• Validation split : StratifiedGroupKFold(5), grouped by patient_index
                     → guarantees zero patient leakage across folds
• Class balancing  : inverse-frequency weights recomputed per training fold
                     and passed to nn.CrossEntropyLoss(weight=...)
• Primary metric   : Teknon-only macro balanced-accuracy
                     (subset of val fold where dataset == "Teknon")
• Output head      : Softmax over 3 logits  (was sigmoid over 1)
• Grad-CAM         : backprop targets each of the 3 classes independently
                     → produces a (3, 12, T_pool) heatmap array

Artifacts written
─────────────────
  E4_cnn_3class_cv_results.csv
  E4_cnn_3class_test_results.csv
  fig_E4_confusion_matrix.png
  fig_E4_gradcam_class_avg.png          ← 3-panel heatmap (one per macroclass)
  fig_E4_gradcam_<label>.png            ← per-patient overlays (3 examples)
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import time
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe; swap to "TkAgg" / remove for GUI
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight

# ──────────────────────────────────────────────────────────────────────────────
# 1.  GLOBAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
N_FOLDS       = 5
N_CLASSES     = 3
MAX_EPOCHS    = 80          # hard ceiling per fold
PATIENCE      = 10          # early-stopping patience (on Teknon bal-acc)
BATCH_SIZE    = 64
LR_INIT       = 1e-3
LR_FACTOR     = 0.5         # ReduceLROnPlateau factor
LR_PATIENCE   = 5           # ReduceLROnPlateau patience
LR_MIN        = 1e-5
L2_REG        = 1e-4
DROPOUT       = 0.3
N_AVG_GRADCAM = 400         # max samples per class for averaged Grad-CAM


LEAD_NAMES  = ["I", "II", "III", "aVR", "aVL", "aVF",
               "V1", "V2", "V3", "V4", "V5", "V6"]
CLASS_NAMES = ["RVOT", "RCC-LCC", "LVOT"]  # indices 0, 1, 2

# File paths — relative to script location (../data/)
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

PATHS = dict(
    X_train     = os.path.join(DATA_DIR, "X_train_pool_task2_3class.npy"),
    y_train     = os.path.join(DATA_DIR, "y_train_pool_task2_3class.npy"),
    info_train  = os.path.join(DATA_DIR, "info_train_pool_task2_3class.csv"),
    X_test      = os.path.join(DATA_DIR, "X_teknon_final_test_task2_3class.npy"),
    y_test      = os.path.join(DATA_DIR, "y_teknon_final_test_task2_3class.npy"),
)

# Device selection: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
if DEVICE.type == "mps":
    os.environ["MPS_FALLBACK_TO_CPU"] = "1"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
print(f"Device: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    """Load all numpy arrays and the training metadata CSV.

    Returns
    -------
    X_train_pool : ndarray  (2935, 12, 200)  float32
    y_train_pool : ndarray  (2935,)           int64
    info_train   : DataFrame with columns [patient_index, dataset, ...]
    X_test       : ndarray  (35, 12, 200)    float32
    y_test       : ndarray  (35,)             int64
    """
    X_train_pool = np.load(PATHS["X_train"]).astype(np.float32)
    y_train_pool = np.load(PATHS["y_train"]).astype(np.int64)
    info_train   = pd.read_csv(PATHS["info_train"])
    X_test       = np.load(PATHS["X_test"]).astype(np.float32)
    y_test       = np.load(PATHS["y_test"]).astype(np.int64)

    print(f"X_train_pool : {X_train_pool.shape}")
    print(f"y_train_pool : {y_train_pool.shape}  class counts: "
          f"{dict(zip(*np.unique(y_train_pool, return_counts=True)))}")
    print(f"X_test       : {X_test.shape}")
    print(f"y_test       : {y_test.shape}   class counts: "
          f"{dict(zip(*np.unique(y_test, return_counts=True)))}")
    return X_train_pool, y_train_pool, info_train, X_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
# 3.  DATA PREPARATION  —  per-sample z-score normalisation
# ──────────────────────────────────────────────────────────────────────────────
def zscore_per_sample(X: np.ndarray) -> np.ndarray:
    """Z-score each (12, 200) matrix independently across all 2400 values.

    Parameters
    ----------
    X : ndarray of shape (N, 12, 200)

    Returns
    -------
    X_z : ndarray of shape (N, 12, 200), float32
    """
    flat = X.reshape(len(X), -1)                       # (N, 2400)
    mu  = flat.mean(axis=1, keepdims=True)
    sig = flat.std(axis=1, keepdims=True) + 1e-8
    X_z = ((flat - mu) / sig).reshape(X.shape)
    return X_z.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PYTORCH DATASET
# ──────────────────────────────────────────────────────────────────────────────
class QRSDataset(Dataset):
    """Wraps (N, 12, 200) ECG tensor and (N,) label array for DataLoader.

    The CNN uses a 1D convolution analogue via Conv2d with kernel (1, k):
    input shape fed to the model is (batch, 1, 12, 200), i.e. a single
    'channel' 2-D image where height = leads and width = time.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        # X: (N, 12, 200) → (N, 1, 12, 200) for Conv2d
        self.X       = torch.from_numpy(X[:, np.newaxis, :, :])   # float32
        self.y       = torch.from_numpy(y)                         # int64
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]   # (1, 12, 200)
        y = self.y[idx]

        if self.augment:
            # Gaussian noise: sigma ~ Uniform(0.01, 0.05)
            sigma = float(torch.empty(1).uniform_(0.01, 0.05))
            x = x + torch.randn_like(x) * sigma

            # Per-sample amplitude scaling ~ Uniform(0.85, 1.15)
            amp = float(torch.empty(1).uniform_(0.85, 1.15))
            x = x * amp

            # Per-lead amplitude jitter ~ Uniform(0.95, 1.05): shape (1, 12, 1)
            lead_jit = torch.empty(1, 12, 1).uniform_(0.95, 1.05)
            x = x * lead_jit

        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CNN ARCHITECTURE  —  Saglietto-style lead-aware 1D CNN (3-class edition)
# ──────────────────────────────────────────────────────────────────────────────
class LeadAwareCNN3Class(nn.Module):
    """Saglietto-mirror 1D CNN for 3-class SOO classification.

    Design rules (strictly enforced)
    ─────────────────────────────────
    (a) Temporal-only stage:  Conv2d with kernel (1, k) — slides along time
        within each lead; leads are NEVER mixed here.
    (b) Spatial fusion stage: 1×1 Conv2d mixes across channels (lead features)
        at each (lead, time) position. This layer is the Grad-CAM target.
    (c) Global average pooling collapses the (lead, time) spatial dims into
        a fixed-length feature vector — leads first fused here.
    (d) Dense classification head → Softmax over 3 classes.

    Input  : (batch, 1, 12, 200)   [channels=1, height=leads, width=time]
    Output : (batch, 3)            [log-probabilities, raw logits]

    Note: nn.CrossEntropyLoss expects raw logits (no explicit softmax needed),
    so the forward pass returns logits. Softmax is applied at inference time.
    """

    def __init__(self, n_classes: int = 3, l2_reg: float = L2_REG,
                 dropout: float = DROPOUT):
        super().__init__()

        # ── Temporal extraction blocks (kernel slides along time only) ──────
        # (1, k) kernels: height dimension = 1 → never mixes leads.
        # 4 blocks with progressive downsampling via MaxPool along time axis.
        #
        # Block 1: (1,  7) conv  → 16 channels → pool (1, 2)  : T 200 → 100
        # Block 2: (1,  5) conv  → 32 channels → pool (1, 2)  : T 100 → 50
        # Block 3: (1,  3) conv  → 32 channels → pool (1, 2)  : T  50 → 25
        # Block 4: (1,  3) conv  → 64 channels → pool (1, 2)  : T  25 → 12
        self.temporal_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1,  16, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            # Block 3
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            # Block 4
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # ── Spatial fusion: 1×1 conv mixes channels per (lead, time) cell ───
        # This is the layer targeted by Grad-CAM (named self.fusion_conv).
        self.fusion_conv = nn.Conv2d(64, 32, kernel_size=(1, 1), bias=False)
        self.fusion_bn   = nn.BatchNorm2d(32)
        # NOTE: inplace=False is required — PyTorch's backward hooks on this
        # layer cannot co-exist with in-place operations (Grad-CAM target).
        self.fusion_act  = nn.LeakyReLU(0.1, inplace=False)  # ← Grad-CAM hook

        # ── Classification head ─────────────────────────────────────────────
        # Global average pooling over (lead, time) → 32-dim vector.
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))   # → (batch, 32, 1, 1)
        self.drop1 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(32, 16)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(16, n_classes)      # raw logits; no softmax

        # Weight decay will be applied by the optimizer via param groups.
        # (L2 regularisation is handled outside the module in PyTorch style.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, 12, 200)  →  logits: (batch, 3)"""
        x = self.temporal_blocks(x)           # → (batch, 64, 12, ~12)
        x = self.fusion_conv(x)               # → (batch, 32, 12, ~12)
        x = self.fusion_bn(x)
        x = self.fusion_act(x)                # ← Grad-CAM target activations
        x = self.gap(x)                       # → (batch, 32,  1,  1)
        x = x.view(x.size(0), -1)             # → (batch, 32)
        x = self.drop1(x)
        x = self.relu1(self.fc1(x))           # → (batch, 16)
        x = self.drop2(x)
        logits = self.fc2(x)                  # → (batch, 3)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def compute_fold_class_weights(y_train_fold: np.ndarray,
                               n_classes: int = N_CLASSES) -> torch.Tensor:
    """Compute inverse-frequency class weights for the training fold.

    Uses sklearn's 'balanced' strategy:  w_c = N / (n_classes * count_c)
    Returns a float32 Tensor of shape (n_classes,) on CPU (moved to device
    inside the training loop).
    """
    classes = np.arange(n_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_fold,
    )
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device) -> float:
    """Train for one epoch. Returns mean training loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module,
             X: np.ndarray,
             y: np.ndarray,
             device: torch.device,
             batch_size: int = 256) -> dict:
    """Run inference and compute multi-class metrics.

    Parameters
    ----------
    model      : trained LeadAwareCNN3Class
    X          : ndarray (N, 12, 200)
    y          : ndarray (N,) int64
    device     : torch device
    batch_size : mini-batch size for inference

    Returns
    -------
    dict with keys: accuracy, balanced_accuracy, macro_f1, weighted_f1,
                    macro_roc_auc (OvR), probas (N, 3), preds (N,)
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    dataset = QRSDataset(X, y, augment=False)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=(device.type == "cuda"))
    all_logits = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        all_logits.append(model(X_batch).cpu())
    logits = torch.cat(all_logits, dim=0)             # (N, 3)
    probas = torch.softmax(logits, dim=1).numpy()     # (N, 3)
    preds  = probas.argmax(axis=1)

    acc    = accuracy_score(y, preds)
    bal    = balanced_accuracy_score(y, preds)
    f1_mac = f1_score(y, preds, average="macro",    zero_division=0)
    f1_wt  = f1_score(y, preds, average="weighted", zero_division=0)

    # Macro OvR AUC — falls back gracefully if a class is absent in y
    try:
        auc = roc_auc_score(y, probas, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return dict(
        accuracy=acc,
        balanced_accuracy=bal,
        macro_f1=f1_mac,
        weighted_f1=f1_wt,
        macro_roc_auc=auc,
        probas=probas,
        preds=preds,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CROSS-VALIDATION LOOP  —  StratifiedGroupKFold, anti-patient-leakage
# ──────────────────────────────────────────────────────────────────────────────
def run_cv(X_train_pool: np.ndarray,
           y_train_pool: np.ndarray,
           info_train: pd.DataFrame) -> tuple:
    """5-fold StratifiedGroupKFold CV with anti-patient-leakage guarantee.

    Folding groups   : info_train["patient_index"]
    Stratification   : y_train_pool (class labels)
    Primary metric   : Teknon-only macro balanced-accuracy within each fold
    Class weights    : recomputed from the training fold's label distribution
    Early stopping   : based on Teknon-subset balanced-accuracy (patience=10)

    Returns
    -------
    cv_df        : DataFrame with per-fold metrics
    best_model   : model from the fold with highest Teknon balanced-accuracy
    fold_models  : list of all 5 trained models (best-epoch weights)
    """
    sgkf   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                   random_state=RANDOM_STATE)
    groups = info_train["patient_index"].values

    # Identify Teknon indices in the overall training pool
    is_teknon_mask = (info_train["dataset"].str.strip() == "teknon").values

    cv_rows     = []
    fold_models = []
    best_cv_bal = -1.0
    best_model  = None

    t0 = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(
            sgkf.split(X_train_pool, y_train_pool, groups)):

        print(f"\n{'─'*60}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}  "
              f"(train={len(tr_idx)}, val={len(va_idx)})")

        # ── Slice training and validation sets ──────────────────────────────
        X_tr, y_tr = X_train_pool[tr_idx], y_train_pool[tr_idx]
        X_va, y_va = X_train_pool[va_idx], y_train_pool[va_idx]

        # Teknon-only validation slice (may be 0 rows in some folds)
        tek_mask_va = is_teknon_mask[va_idx]
        X_tek = X_va[tek_mask_va]
        y_tek = y_va[tek_mask_va]
        has_teknon_val = len(y_tek) > 0

        print(f"  Train class dist : {dict(zip(*np.unique(y_tr, return_counts=True)))}")
        print(f"  Val   class dist : {dict(zip(*np.unique(y_va, return_counts=True)))}")
        if has_teknon_val:
            print(f"  Teknon val subset: {len(y_tek)} samples  "
                  f"{dict(zip(*np.unique(y_tek, return_counts=True)))}")
        else:
            print("  Teknon val subset: none in this fold — using global val bal-acc")

        # ── Class weights from TRAINING fold only ───────────────────────────
        class_weights = compute_fold_class_weights(y_tr).to(DEVICE)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)

        # ── Build model + optimizer ─────────────────────────────────────────
        torch.manual_seed(RANDOM_STATE + fold_idx)
        model = LeadAwareCNN3Class(n_classes=N_CLASSES,
                                   l2_reg=L2_REG,
                                   dropout=DROPOUT).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(),
                                lr=LR_INIT,
                                weight_decay=L2_REG)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=LR_FACTOR,
            patience=LR_PATIENCE, min_lr=LR_MIN
        )

        # ── DataLoaders ──────────────────────────────────────────────────────
        tr_dataset = QRSDataset(X_tr, y_tr, augment=True)
        tr_loader  = DataLoader(tr_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0,
                                pin_memory=(DEVICE.type == "cuda"))

        # ── Training loop with early stopping ───────────────────────────────
        best_epoch_bal  = -1.0
        best_epoch_wts  = None
        epochs_no_imprv = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, tr_loader, criterion,
                                         optimizer, DEVICE)

            # Primary metric: Teknon-subset balanced-accuracy if available,
            # otherwise fall back to global validation balanced-accuracy.
            if has_teknon_val:
                metrics_es = evaluate(model, X_tek, y_tek, DEVICE)
            else:
                metrics_es = evaluate(model, X_va,  y_va,  DEVICE)
            es_bal = metrics_es["balanced_accuracy"]

            scheduler.step(es_bal)

            if es_bal > best_epoch_bal:
                best_epoch_bal = es_bal
                best_epoch_wts = copy.deepcopy(model.state_dict())
                epochs_no_imprv = 0
            else:
                epochs_no_imprv += 1

            if epoch % 10 == 0 or epochs_no_imprv == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:3d}  loss={train_loss:.4f}  "
                      f"tek_bal={es_bal:.4f}  best={best_epoch_bal:.4f}  "
                      f"lr={lr_now:.2e}", flush=True)

            if epochs_no_imprv >= PATIENCE:
                print(f"  ← Early stop at epoch {epoch} "
                      f"(best={best_epoch_bal:.4f})", flush=True)
                break

        # Restore best weights for this fold
        model.load_state_dict(best_epoch_wts)
        fold_models.append(model)

        # ── Fold-level evaluation (full validation set) ──────────────────────
        val_metrics  = evaluate(model, X_va, y_va, DEVICE)
        tek_metrics  = (evaluate(model, X_tek, y_tek, DEVICE)
                        if has_teknon_val
                        else {"balanced_accuracy": float("nan"),
                              "macro_f1": float("nan")})

        row = dict(
            fold                = fold_idx,
            val_accuracy        = val_metrics["accuracy"],
            val_balanced_acc    = val_metrics["balanced_accuracy"],
            val_macro_f1        = val_metrics["macro_f1"],
            val_roc_auc         = val_metrics["macro_roc_auc"],
            tek_balanced_acc    = tek_metrics["balanced_accuracy"],
            tek_macro_f1        = tek_metrics["macro_f1"],
            best_epoch_bal_acc  = best_epoch_bal,
            epochs_trained      = epoch,
            n_val               = len(y_va),
            n_tek_val           = len(y_tek),
        )
        cv_rows.append(row)

        print(f"  Fold result: val_bal={val_metrics['balanced_accuracy']:.4f}  "
              f"tek_bal={tek_metrics['balanced_accuracy']:.4f}  "
              f"val_auc={val_metrics['macro_roc_auc']:.4f}")

        # Track best fold model by Teknon balanced-accuracy
        criterion_score = (tek_metrics["balanced_accuracy"]
                           if has_teknon_val
                           else val_metrics["balanced_accuracy"])
        if criterion_score > best_cv_bal:
            best_cv_bal = criterion_score
            best_model  = copy.deepcopy(model)

    elapsed = time.time() - t0
    print(f"\nCV finished in {elapsed:.0f}s ({elapsed/60:.1f} min).")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv("E4_cnn_3class_cv_results.csv", index=False)

    print("\n=== Per-fold CV results ===")
    numeric_cols = [c for c in cv_df.columns
                    if c not in ("fold", "epochs_trained", "n_val", "n_tek_val")]
    print(cv_df[["fold"] + numeric_cols + ["epochs_trained"]].round(4).to_string(index=False))

    print("\n=== CV summary (mean ± std) ===")
    for col in numeric_cols:
        vals = cv_df[col].dropna()
        if len(vals):
            print(f"  {col:25s}: {vals.mean():.4f} ± {vals.std():.4f}")

    return cv_df, best_model, fold_models


# ──────────────────────────────────────────────────────────────────────────────
# 8.  FINAL MODEL  —  retrain on entire pool, evaluate on Teknon held-out test
# ──────────────────────────────────────────────────────────────────────────────
def train_final_model(X_train_pool: np.ndarray,
                      y_train_pool: np.ndarray) -> nn.Module:
    """Retrain a fresh CNN on the full training pool.

    Uses a stratified 10 % held-out split for early stopping
    (same protocol as the binary notebook §7).
    """
# Enforce a strict 10% stratified split so early stopping sees the true minority proportions
    skf_final = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    tr_idx, es_idx = next(skf_final.split(X_train_pool, y_train_pool))

    X_tr, y_tr = X_train_pool[tr_idx], y_train_pool[tr_idx]
    X_es, y_es = X_train_pool[es_idx], y_train_pool[es_idx]
    print(f"Final retraining: {len(X_tr)} train / {len(X_es)} early-stop")

    class_weights = compute_fold_class_weights(y_tr).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    torch.manual_seed(RANDOM_STATE)
    model = LeadAwareCNN3Class(n_classes=N_CLASSES,
                               l2_reg=L2_REG,
                               dropout=DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=L2_REG)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN
    )

    tr_dataset = QRSDataset(X_tr, y_tr, augment=True)
    tr_loader  = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0,
                            pin_memory=(DEVICE.type == "cuda"))

    best_bal   = -1.0
    best_wts   = None
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, tr_loader, criterion,
                                     optimizer, DEVICE)
        es_metrics = evaluate(model, X_es, y_es, DEVICE)
        es_bal     = es_metrics["balanced_accuracy"]
        scheduler.step(es_bal)

        if es_bal > best_bal:
            best_bal  = es_bal
            best_wts  = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve == 0:
            print(f"  Epoch {epoch:3d}  loss={train_loss:.4f}  "
                  f"es_bal={es_bal:.4f}  best={best_bal:.4f}", flush=True)
        if no_improve >= PATIENCE:
            print(f"  ← Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_wts)
    print(f"Final model trained in {time.time()-t0:.0f}s, {epoch} epochs.")
    return model


def evaluate_final(model: nn.Module,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> dict:
    """Full evaluation on the Teknon held-out test set."""
    metrics = evaluate(model, X_test, y_test, DEVICE)

    pd.DataFrame([{k: v for k, v in metrics.items()
                   if k not in ("probas", "preds")}]).to_csv(
        "E4_cnn_3class_test_results.csv", index=False)

    print("\n=== Teknon held-out test results (3-class CNN) ===")
    for k, v in metrics.items():
        if k not in ("probas", "preds"):
            print(f"  {k:25s}: {v:.4f}")
    print()
    print(classification_report(
        y_test, metrics["preds"],
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 9.  CONFUSION MATRIX FIGURE
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          save_path: str = "fig_E4_confusion_matrix.png") -> None:
    """Plot and save a 3×3 confusion matrix."""
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("3-class SOO CNN — Teknon held-out test", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 10.  MULTI-CLASS GRAD-CAM IMPLEMENTATION
# ──────────────────────────────────────────────────────────────────────────────
def compute_gradcam(model: nn.Module,
                    X: np.ndarray,
                    target_class: int,
                    device: torch.device = DEVICE,
                    batch_size: int = 64) -> np.ndarray:
    """Compute Grad-CAM heatmaps for a single target class over all samples.

    The gradient is computed with respect to the activations of
    model.fusion_act (the LeakyReLU immediately after the 1×1 fusion conv),
    which is the spatial-temporal bottleneck just before GlobalAveragePooling.

    Algorithm (Selvaraju et al. 2017)
    ──────────────────────────────────
    1. Forward pass → record fusion_act activations A (batch, C, H, W)
    2. Backprop the *target-class logit* (score_{target_class}) → gradients G
    3. Global-average-pool gradients over (H, W): α_c = mean_{h,w}(G_c)
    4. Weighted sum of activation maps: L = ReLU(Σ_c α_c * A_c)
    5. Normalise each heatmap to [0, 1] per sample

    Parameters
    ----------
    model        : trained LeadAwareCNN3Class (eval mode)
    X            : ndarray (N, 12, 200) — samples to explain
    target_class : int in {0, 1, 2}
    device       : torch.device

    Returns
    -------
    heatmaps : ndarray (N, 12, T_pool)
        ReLU-and-max-normalised Grad-CAM maps; T_pool ≈ 12 after 4× downsampling.
        Axis 1 = leads, axis 2 = pooled time positions.
    """
    model.eval()

    # Storage containers for hook data (filled per forward/backward pass)
    _activations: list[torch.Tensor] = []
    _gradients:   list[torch.Tensor] = []

    def _fwd_hook(module, input, output):
        _activations.append(output.detach().clone())

    def _bwd_hook(module, grad_input, grad_output):
        _gradients.append(grad_output[0].detach().clone())

    fwd_handle = model.fusion_act.register_forward_hook(_fwd_hook)
    bwd_handle = model.fusion_act.register_full_backward_hook(_bwd_hook)

    all_heatmaps = []
    dataset = QRSDataset(X, np.zeros(len(X), dtype=np.int64), augment=False)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0)

    try:
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            X_batch.requires_grad_(False)

            _activations.clear()
            _gradients.clear()

            # Forward — track computation graph for this specific class score
            logits = model(X_batch)                          # (B, 3)
            score  = logits[:, target_class].sum()           # scalar

            # Backward from that class's summed score
            model.zero_grad()
            score.backward()

            acts  = _activations[0]   # (B, C, H, W)  where H=leads, W=time_pool
            grads = _gradients[0]     # (B, C, H, W)

            # α_c = mean gradient over (H, W) for each channel
            alphas = grads.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)

            # Weighted activation maps
            cam = (alphas * acts).sum(dim=1)                 # (B, H, W)
            cam = torch.relu(cam)                            # ReLU

            # Normalise per sample (max → 1)
            cam_np = cam.cpu().numpy()
            max_v  = cam_np.max(axis=(1, 2), keepdims=True)
            cam_np = np.divide(cam_np, max_v,
                               out=np.zeros_like(cam_np),
                               where=max_v > 0)
            all_heatmaps.append(cam_np)

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    return np.concatenate(all_heatmaps, axis=0)    # (N, 12, T_pool)


# ──────────────────────────────────────────────────────────────────────────────
# 11.  CLASS-AVERAGED GRAD-CAM HEATMAP  (3-panel figure)
# ──────────────────────────────────────────────────────────────────────────────
def compute_and_plot_class_avg_gradcam(
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        preds: np.ndarray,
        save_path: str = "fig_E4_gradcam_class_avg.png",
        n_avg: int = N_AVG_GRADCAM) -> np.ndarray:
    """Compute class-averaged Grad-CAM for all 3 macroclasses.

    For each class c:
      1. Collect up to `n_avg` correctly-classified samples of class c.
      2. Compute Grad-CAM maps targeting class c.
      3. Average across samples → (12, T_pool) heatmap.

    Returns
    -------
    cam_avg : ndarray (3, 12, T_pool)
        Stacked class-averaged heatmaps (class-axis first).
    """
    correct = (preds == y)
    rng     = np.random.RandomState(RANDOM_STATE)
    cams    = []

    for cls_idx in range(N_CLASSES):
        mask = correct & (y == cls_idx)
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            print(f"  WARNING: no correctly-classified samples for class "
                  f"{cls_idx} ({CLASS_NAMES[cls_idx]}) — using all samples.")
            idx = np.where(y == cls_idx)[0]
        idx = rng.choice(idx, size=min(n_avg, len(idx)), replace=False)
        print(f"  Grad-CAM class {cls_idx} ({CLASS_NAMES[cls_idx]}): "
              f"{len(idx)} samples", flush=True)
        cam_c = compute_gradcam(model, X[idx], target_class=cls_idx)
        cams.append(cam_c.mean(axis=0))      # (12, T_pool)

    cam_avg = np.stack(cams, axis=0)         # (3, 12, T_pool)

    # ── Plot ─────────────────────────────────────────────────────────────────
    T_pool = cam_avg.shape[2]
    xticks = np.linspace(0, T_pool - 1, 11)
    xtick_labels = [f"{p}%" for p in range(0, 101, 10)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, cam, cname in zip(axes, cam_avg, CLASS_NAMES):
        vmax = cam.max() if cam.max() > 0 else 1.0
        im = ax.imshow(cam, aspect="auto", cmap="viridis",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(len(LEAD_NAMES)))
        ax.set_yticklabels(LEAD_NAMES, fontsize=9)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("QRS time (% of window, post-pooling)", fontsize=9)
        ax.set_title(f"Grad-CAM — {cname}\n"
                     f"(avg. over correctly-classified samples)",
                     fontsize=10)
        fig.colorbar(im, ax=ax, label="Activation", fraction=0.046, pad=0.04)

    axes[0].set_ylabel("ECG Lead", fontsize=10)
    plt.suptitle("3-class SOO CNN — Per-class Grad-CAM Heatmaps",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")

    return cam_avg


# ──────────────────────────────────────────────────────────────────────────────
# 12.  PER-PATIENT GRAD-CAM OVERLAY  (3 example figures)
# ──────────────────────────────────────────────────────────────────────────────
def plot_patient_gradcam(sample_idx: int,
                         model: nn.Module,
                         X: np.ndarray,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         probas: np.ndarray,
                         target_class: int | None = None,
                         title_extra: str = "") -> plt.Figure:
    """Overlay Grad-CAM heatmap on raw 12-lead QRS for one sample.

    Parameters
    ----------
    sample_idx   : index into X / y arrays
    model        : trained CNN
    X            : ndarray (N, 12, 200)  raw signals
    y_true       : ndarray (N,) int labels
    y_pred       : ndarray (N,) int predicted labels
    probas       : ndarray (N, 3) softmax probabilities
    target_class : class to explain; defaults to predicted class
    title_extra  : optional extra string appended to suptitle

    Returns
    -------
    matplotlib Figure
    """
    if target_class is None:
        target_class = int(y_pred[sample_idx])

    # Compute Grad-CAM for this single sample
    cam = compute_gradcam(model, X[sample_idx:sample_idx+1],
                          target_class=target_class)[0]   # (12, T_pool)

    # Upsample temporal dimension to match original 200 timepoints (nearest)
    T_raw  = X.shape[2]
    T_pool = cam.shape[1]
    repeat = T_raw // T_pool + 1
    cam_up = np.repeat(cam, repeat, axis=1)[:, :T_raw]    # (12, 200)

    true_lab = CLASS_NAMES[int(y_true[sample_idx])]
    pred_lab = CLASS_NAMES[int(y_pred[sample_idx])]
    correct  = "✓" if true_lab == pred_lab else "✗"
    prob_str = "  ".join(f"P({n})={probas[sample_idx, i]:.2f}"
                         for i, n in enumerate(CLASS_NAMES))

    fig, axes = plt.subplots(3, 4, figsize=(14, 6), sharex=True)
    t = np.arange(T_raw)
    for li, (ax, lead) in enumerate(zip(axes.flat, LEAD_NAMES)):
        y_min = float(X[sample_idx, li, :].min())
        y_max = float(X[sample_idx, li, :].max())
        ax.imshow(cam_up[li:li+1, :], aspect="auto", cmap="Reds",
                  alpha=0.45, vmin=0, vmax=1,
                  extent=[0, T_raw - 1, y_min, y_max])
        ax.plot(t, X[sample_idx, li, :], color="k", lw=1.0)
        ax.set_title(lead, fontsize=9)
        ax.grid(alpha=0.3)

    title = (f"True={true_lab}  Pred={pred_lab} {correct}  |  "
             f"{prob_str}  |  {title_extra}")
    plt.suptitle(title, y=1.00, fontsize=10)
    plt.tight_layout()
    return fig


def save_example_gradcam_figures(model: nn.Module,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray,
                                 test_metrics: dict) -> None:
    """Find and save one correct example per class + one misclassification."""
    preds  = test_metrics["preds"]
    probas = test_metrics["probas"]

    examples = []

    # One correctly-classified sample per class
    for cls_idx, cname in enumerate(CLASS_NAMES):
        correct_idx = np.where((y_test == cls_idx) & (preds == cls_idx))[0]
        if len(correct_idx):
            examples.append((f"correct_{cname.replace('-','_')}",
                             correct_idx[0], cls_idx))

    # One misclassification (any class)
    error_idx = np.where(y_test != preds)[0]
    if len(error_idx):
        examples.append(("error", error_idx[0], None))

    for label, idx, tgt_cls in examples:
        fig = plot_patient_gradcam(
            idx, model, X_test, y_test, preds, probas,
            target_class=tgt_cls,
            title_extra=f"sample #{idx}",
        )
        fname = f"fig_E4_gradcam_{label}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# 13.  MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("E4 — 3-class SOO CNN (PyTorch, Saglietto-mirror)")
    print("=" * 60)
    print(f"Parameters: RANDOM_STATE={RANDOM_STATE}  N_FOLDS={N_FOLDS}  "
          f"MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}  "
          f"BATCH_SIZE={BATCH_SIZE}  LR={LR_INIT}")
    print(f"Model params: ~{count_parameters(LeadAwareCNN3Class()):,} trainable")
    print()

    # ── 2. Load ──────────────────────────────────────────────────────────────
    X_train_pool, y_train_pool, info_train, X_test, y_test = load_data()

    # ── 3. Normalise ─────────────────────────────────────────────────────────
    print("\nApplying per-sample z-score normalisation …")
    X_train_pool = zscore_per_sample(X_train_pool)
    X_test       = zscore_per_sample(X_test)
    print(f"  Train range after norm: "
          f"[{X_train_pool.min():.3f}, {X_train_pool.max():.3f}]")

    # ── 7. Cross-validation ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION  (StratifiedGroupKFold, anti-patient-leakage)")
    print("=" * 60)
    cv_df, best_cv_model, fold_models = run_cv(
        X_train_pool, y_train_pool, info_train
    )

    # ── 8. Final model ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL MODEL  —  retrain on full pool, evaluate on Teknon test")
    print("=" * 60)
    final_model  = train_final_model(X_train_pool, y_train_pool)
    test_metrics = evaluate_final(final_model, X_test, y_test)
	# SAVE WEIGHTS FOR DIAGNOSTIC SCRIPTS
    torch.save(final_model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "E4_final_model_weights.pth"))
    print("[SAVED] Exported E4_final_model_weights.pth successfully.")

# ==============================================================================
    # EXPERIMENT 3 PRODUCTION VISUALIZATION SUITE (CORRECTED)
    # ==============================================================================

    y_test_np = np.asarray(y_test)
    preds_np = np.asarray(test_metrics["preds"])
    scores_np = np.asarray(test_metrics["probas"])

    # ── 1. EXPORT COMBINED CONFUSION MATRIX & MULTI-CLASS ROC CURVES
    print("\n  [Processing] Rendering fig_exp3_confusion_roc.png...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Confusion Matrix
    cm = confusion_matrix(y_test_np, preds_np, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=axes[0], cmap="Blues", colorbar=True, values_format="d")
    axes[0].set_title("Confusion Matrix (Held-out Teknon Test)", fontsize=11, fontweight="bold", pad=10)

    # Panel B: One-vs-Rest ROC Curves
    y_test_binarized = label_binarize(y_test_np, classes=[0, 1, 2])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], scores_np[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1].plot(
            fpr, tpr, color=colors[i], lw=2,
            label=f"ROC {CLASS_NAMES[i]} (AUC = {roc_auc:.2f})"
        )

    axes[1].plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate", fontsize=10)
    axes[1].set_ylabel("True Positive Rate", fontsize=10)
    axes[1].set_title("One-vs-Rest ROC Curves (1D CNN Head)", fontsize=11, fontweight="bold", pad=10)
    axes[1].legend(loc="lower right", fontsize=9, frameon=True)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig_exp3_confusion_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [Saved] fig_exp3_confusion_roc.png")

    # ── 2. MULTI-CLASS GRAD-CAM AVERAGED HEATMAP PLOT
    print("\n" + "=" * 60)
    print("GRAD-CAM  —  class-averaged heatmaps")
    print("=" * 60)

    # Re-evaluate training pool for the averaged heatmaps
    train_metrics_for_cam = evaluate(final_model, X_train_pool, y_train_pool, DEVICE)
    cam_avg = compute_and_plot_class_avg_gradcam(
        final_model,
        X_train_pool,
        y_train_pool,
        train_metrics_for_cam["preds"],
    )
    np.save("E4_gradcam_class_avg.npy", cam_avg)

    # Generate polished production plot for Grad-CAM
    fig_cam, axes_cam = plt.subplots(1, N_CLASSES, figsize=(5 * N_CLASSES, 4))
    for c in range(N_CLASSES):
        im = axes_cam[c].imshow(cam_avg[c], aspect="auto", cmap="jet", interpolation="nearest")
        axes_cam[c].set_title(f"Grad-CAM Attention: {CLASS_NAMES[c]}", fontsize=10, fontweight="bold")
        axes_cam[c].set_xlabel("Time Bins", fontsize=9)
        if c == 0:
            axes_cam[c].set_ylabel("ECG Leads (1-12)", fontsize=9)
        fig_cam.colorbar(im, ax=axes_cam[c], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("fig_exp3_gradcam.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cam)
    print("  [Saved] fig_exp3_gradcam.png & E4_gradcam_class_avg.npy")

    # ── 3. EXPORT CNN VS CLASSICAL ML BENCHMARK COMPARISON PLOT
    print("\n  [Processing] Compiling comparative metrics chart...")
    
    # Extract deep learning performance metrics
    cnn_bal_acc = test_metrics["balanced_accuracy"]
    cnn_macro_f1 = test_metrics["macro_f1"]
    
    # Read the classical model metrics from Experiment 1 fallback file
    rf_bal_acc, rf_macro_f1 = 0.5299, 0.5210  # Fallback to your winning Exp 1 results
    if os.path.exists("task2_final_test_results.csv"):
        try:
            rf_df = pd.read_csv("task2_final_test_results.csv")
            if "balanced_accuracy" in rf_df.columns and "macro_f1" in rf_df.columns:
                rf_bal_acc = rf_df.iloc[0]["balanced_accuracy"]
                rf_macro_f1 = rf_df.iloc[0]["macro_f1"]
        except Exception:
            pass

    # Render a side-by-side grouped bar chart comparing performance architectures
    metrics_labels = ["Balanced Accuracy", "Macro F1-Score"]
    rf_scores = [rf_bal_acc, rf_macro_f1]
    cnn_scores = [cnn_bal_acc, cnn_macro_f1]
    
    x = np.arange(len(metrics_labels))
    width = 0.35

    fig_comp, ax_comp = plt.subplots(figsize=(7, 5))
    rects1 = ax_comp.bar(x - width/2, rf_scores, width, label="Random Forest (Exp 1)", color="#B0C4DE")
    rects2 = ax_comp.bar(x + width/2, cnn_scores, width, label="Lead-Aware 1D CNN (Exp 3)", color="#4682B4")

    ax_comp.set_ylabel("Evaluation Scores (0.0 - 1.0)", fontsize=10)
    ax_comp.set_title("Architecture Comparison: Classical ML vs. Deep Learning", fontsize=11, fontweight="bold", pad=12)
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(metrics_labels, fontsize=10)
    ax_comp.set_ylim([0.0, 1.0])
    ax_comp.legend(loc="upper right", frameon=True)
    ax_comp.grid(axis="y", alpha=0.3)

    # Attach numerical indicators directly over the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax_comp.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig("fig_exp3_cnn_vs_rf.png", dpi=150, bbox_inches="tight")
    plt.close(fig_comp)
    print("  [Saved] fig_exp3_cnn_vs_rf.png")

    print("\n" + "=" * 60)
    print("Execution Completed Successfully. All Experiment 3 Figures Saved.")
    print("=" * 60)

if __name__ == "__main__":
    main()