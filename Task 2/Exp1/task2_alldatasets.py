"""
================================================================================
TASK 2 — QRS Morphology Pipeline: 3-Class Site-of-Origin Classification
================================================================================

Goal:
    Reproduce and extend the QRS-morphology approach of Doste et al. (2022) and
    Bocanegra-Pérez et al. (2024) for a 3-CLASS site-of-origin classification:
        Class 0 → RVOT  (RVOT Septum & RVOT Free Wall)
        Class 1 → RCC-LCC (RCC-LCC-Commissure)
        Class 2 → LVOT  (LVOT Summit & LVOT Subvalvular)

Pipeline structure (strictly sequential — each phase consumes the decision of the last):
    0.  Imports, global config & helpers
    1.  Load data + quick EDA
    2.  Representation   (12 leads × 10 temporal bins = 120 features)
    3.  Phase A — Dataset selection    →  best_dataset_combo
    4.  Phase B — Lead selection       →  best_lead_combo   (uses A)
    5.  Phase C — Model selection      →  best_model        (uses A + B)
    6.  Final evaluation  — train ONCE on full designated pool,
                            evaluate ONCE on held-out Teknon test
    7.  Interpretability  — feature importance / SHAP / lead×bin heatmap
"""

# ==============================================================================
# 0.  IMPORTS, GLOBAL CONFIG & HELPERS
# ==============================================================================

import os
import time
import warnings
from pathlib import Path
from itertools import combinations

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in scripts
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ---- Optional extras ----
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception as _e:
    XGB_AVAILABLE = False
    print(f"[INFO] XGBoost not available ({_e!s}); will be skipped in Phase C.")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception as _e:
    SHAP_AVAILABLE = False
    print(f"[INFO] SHAP not available ({_e!s}); SHAP block will be skipped.")

# -----------------------------------------------------------------------
# Global config
# -----------------------------------------------------------------------
RANDOM_STATE  = 42
N_FOLDS       = 5
N_BINS        = 10

LEAD_NAMES  = ["I", "II", "III", "aVR", "aVL", "aVF",
               "V1", "V2", "V3", "V4", "V5", "V6"]

CLASS_NAMES = ["RVOT", "RCC-LCC", "LVOT"]   # 0=RVOT | 1=RCC-LCC | 2=LVOT
N_CLASSES   = 3

np.random.seed(RANDOM_STATE)

# Replaced StratifiedKFold with StratifiedGroupKFold to prevent internal patient data leakage
SGKF = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# -----------------------------------------------------------------------
# Data file paths
# -----------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR   = _SCRIPT_DIR / ".." / "data"

def find_file(filename: str) -> Path:
    candidates = [
        _DATA_DIR / filename,
        Path(".") / filename,
        Path("..") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"Could not find '{filename}'.")


print("=" * 70)
print("TASK 2 — 3-Class QRS Morphology Pipeline (Patient Group Validated)")
print(f"  N_FOLDS={N_FOLDS} | N_BINS={N_BINS} | N_CLASSES={N_CLASSES}")
print(f"  XGBoost available: {XGB_AVAILABLE} | SHAP available: {SHAP_AVAILABLE}")
print("=" * 70)


# ==============================================================================
# 0.1  DOUBLE-METRIC CROSS-VALIDATION HELPER
# ==============================================================================

def metrics_block(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    out = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1":          f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision":   precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall":      recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if len(np.unique(y_true)) == N_CLASSES and y_score is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        except ValueError:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
    return out


def _scores_from(est, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)
    if hasattr(est, "decision_function"):
        df = est.decision_function(X)
        if df.ndim == 1:
            df = np.column_stack([-df, df])
        return df
    pred = est.predict(X)
    ohe = np.zeros((len(pred), N_CLASSES), dtype=float)
    for i, p in enumerate(pred):
        ohe[i, int(p)] = 1.0
    return ohe


def cv_double_metric(
    model,
    X: np.ndarray,
    y: np.ndarray,
    is_teknon: np.ndarray,
    groups: np.ndarray,
    dataset_train_mask: np.ndarray = None,
    lead_cols: list = None,
    splitter=SGKF,
) -> pd.DataFrame:
    """
    Stratified Group CV protecting against patient leakage while monitoring 
    Global and Teknon-specific validation performance.
    """
    rows = []
    # Pass structural patient groups into the split generation loop
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups=groups)):
        if dataset_train_mask is not None:
            tr = tr[dataset_train_mask[tr]]
        
        if len(tr) == 0:
            continue

        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]

        if lead_cols is not None:
            Xtr = Xtr[:, lead_cols]
            Xva = Xva[:, lead_cols]

        # Explicit handling for algorithmic class weight balancing
        est = clone(model)
        
        # Check if we are running an unwrapped XGBoost model or a Pipeline wrapping it
        is_xgb = False
        clf_step = est
        if isinstance(est, Pipeline):
            clf_step = est.steps[-1][1]
        
        if XGB_AVAILABLE and isinstance(clf_step, XGBClassifier):
            is_xgb = True

        if is_xgb:
            # Manually calculate fold sample weights to force XGBoost to respect minor morphologies
            sw = compute_sample_weight(class_weight="balanced", y=ytr)
            if isinstance(est, Pipeline):
                fit_params = {f"{est.steps[-1][0]}__sample_weight": sw}
                est.fit(Xtr, ytr, **fit_params)
            else:
                est.fit(Xtr, ytr, sample_weight=sw)
        else:
            # Scikit-learn estimators leverage native class_weight internal parameters
            est.fit(Xtr, ytr)

        pred  = est.predict(Xva)
        score = _scores_from(est, Xva)

        g = metrics_block(yva, pred, score)
        tek = is_teknon[va]
        
        if tek.sum() > 0:
            tk = metrics_block(yva[tek], pred[tek], score[tek])
        else:
            tk = {k: np.nan for k in g}

        rows.append({
            "fold": fold,
            "n_train":       len(tr),
            "n_val":         len(va),
            "n_val_teknon":  int(tek.sum()),
            **{f"g_{k}": v for k, v in g.items()},
            **{f"t_{k}": v for k, v in tk.items()},
        })
    return pd.DataFrame(rows)


def summarise(folds_df: pd.DataFrame, label: str) -> dict:
    cols = [c for c in folds_df.columns if c.startswith(("g_", "t_"))]
    agg  = folds_df[cols].agg(["mean", "std"])
    row  = {"config": label}
    for c in cols:
        row[f"{c}_mean"] = agg.loc["mean", c]
        row[f"{c}_std"]  = agg.loc["std",  c]
    return row


print("\n[OK] Patient-blind Double-metric CV helper defined.")


# ==============================================================================
# 1.  LOAD DATA + QUICK EDA
# ==============================================================================

print("\n" + "─" * 70)
print("§1  Loading data …")
print("─" * 70)

X_train_pool = np.load(find_file("X_train_pool_task2_3class.npy"))
y_train_pool = np.load(find_file("y_train_pool_task2_3class.npy")).astype(int)
X_test       = np.load(find_file("X_teknon_final_test_task2_3class.npy"))
y_test       = np.load(find_file("y_teknon_final_test_task2_3class.npy")).astype(int)

info_train   = pd.read_csv(find_file("info_train_pool_task2_3class.csv"))
info_test    = pd.read_csv(find_file("info_teknon_final_test_task2_3class.csv"))

# Extract critical Patient IDs mapping groups to protect against cross-fold contamination

PATIENT_GROUPS = info_train["patient_index"].values
dataset_train = info_train["dataset"].astype(str).values
IS_TEKNON     = (dataset_train == "Teknon")

print(f"\n  X_train_pool : {X_train_pool.shape} | Patient Unique Profiles: {len(np.unique(PATIENT_GROUPS))}")
print(f"  X_test       : {X_test.shape}")


# ==============================================================================
# 2.  REPRESENTATION
# ==============================================================================

print("\n" + "─" * 70)
print("§2  Building 12 × 10 binned QRS features …")
print("─" * 70)

def create_binned_qrs_features(X: np.ndarray, lead_names: list = LEAD_NAMES, n_bins: int = N_BINS, aggregation: str = "mean") -> tuple:
    n_samples, n_leads, n_timepoints = X.shape
    edges = np.linspace(0, n_timepoints, n_bins + 1, dtype=int)
    pct   = np.linspace(0, 100, n_bins + 1, dtype=int)
    Xb    = np.zeros((n_samples, n_leads * n_bins), dtype=np.float32)
    names = []
    for li, lead in enumerate(lead_names):
        for bi in range(n_bins):
            seg = X[:, li, edges[bi] : edges[bi + 1]]
            if aggregation == "mean":
                Xb[:, li * n_bins + bi] = seg.mean(axis=1)
            else:
                Xb[:, li * n_bins + bi] = np.max(np.abs(seg), axis=1)
            names.append(f"{lead}_{pct[bi]}_{pct[bi + 1]}")
    return Xb, names


X_train_binned, FEATURE_NAMES = create_binned_qrs_features(X_train_pool)
X_test_binned,  _             = create_binned_qrs_features(X_test)

LEAD_TO_COLS = {lead: list(range(i * N_BINS, (i + 1) * N_BINS)) for i, lead in enumerate(LEAD_NAMES)}


# ==============================================================================
# 3.  PHASE A — DATASET SELECTION
# ==============================================================================

print("\n" + "=" * 70)
print("PHASE A — Dataset Selection")
print("=" * 70)

DATASET_COMBOS = {
    "A_Sims_only":           ["Sims"],
    "B_real_nonTeknon":      ["CARTO", "Database"],
    "C_Teknon_only":         ["Teknon"],
    "D_Sims_CARTO_Database": ["Sims", "CARTO", "Database"],
    "E_realAll_Teknon":      ["CARTO", "Database", "Teknon"],
    "F_all":                 ["Sims", "CARTO", "Database", "Teknon"],
}

REF_MODEL = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

phaseA_rows = []
t0 = time.time()


def cv_double_metric_weighted(
    model, X, y, is_teknon, groups,
    sample_weights: np.ndarray,   # replaces dataset_train_mask
    lead_cols=None, splitter=SGKF,
):
    rows = []
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups=groups)):
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]
        sw = sample_weights[tr]    # slice weights to this fold's training indices

        if lead_cols is not None:
            Xtr = Xtr[:, lead_cols]
            Xva = Xva[:, lead_cols]

        est = clone(model)

        # Pass sample_weight to any sklearn-compatible estimator
        is_xgb = isinstance(
            est.steps[-1][1] if isinstance(est, Pipeline) else est,
            XGBClassifier if XGB_AVAILABLE else type(None)
        )
        if is_xgb:
            fit_params = {"sample_weight": sw}
            if isinstance(est, Pipeline):
                fit_params = {f"{est.steps[-1][0]}__sample_weight": sw}
            est.fit(Xtr, ytr, **fit_params)
        else:
            try:
                est.fit(Xtr, ytr, sample_weight=sw)
            except TypeError:
                est.fit(Xtr, ytr)   # fallback for estimators that don't accept sw

        pred = est.predict(Xva)
        score = _scores_from(est, Xva)

        g = metrics_block(yva, pred, score)
        tek = is_teknon[va]
        tk = metrics_block(yva[tek], pred[tek], score[tek]) if tek.sum() > 0 \
             else {k: np.nan for k in g}

        rows.append({
            "fold": fold,
            "n_train": len(tr),
            "alpha_mean_nontek": float(sw[~is_teknon[tr]].mean()) if (~is_teknon[tr]).sum() > 0 else 0.0,
            **{f"g_{k}": v for k, v in g.items()},
            **{f"t_{k}": v for k, v in tk.items()},
        })
    return pd.DataFrame(rows)

def build_teknon_anchored_weights(
    dataset_labels: np.ndarray,
    y: np.ndarray,
    alpha: float,           # weight assigned to non-Teknon samples (0=discard, 1=equal)
    class_balance: bool = True,
) -> np.ndarray:
    """
    Assign per-sample weights:
      - Teknon samples: base weight from class balancing
      - Non-Teknon samples: same base weight × alpha

    alpha=0.0  → Teknon-only (current C_Teknon_only)
    alpha=1.0  → all sources equal weight (current F_all)
    alpha=0.1-0.3 → Teknon-anchored with foreign diversity bonus
    """
    is_tek = (dataset_labels == "Teknon").astype(float)
    domain_weights = np.where(is_tek, 1.0, alpha)

    if class_balance:
        class_w = compute_sample_weight("balanced", y=y)
        combined = class_w * domain_weights
    else:
        combined = domain_weights

    # Normalize so Teknon samples retain unit mean weight
    tek_mask = is_tek.astype(bool)
    combined[tek_mask] /= combined[tek_mask].mean()
    combined[~tek_mask] /= (combined[~tek_mask].mean() + 1e-9)
    combined[~tek_mask] *= alpha   # re-apply alpha after per-class normalization

    return combined


# Phase A becomes a 1D search over alpha
ALPHA_GRID = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0]
# alpha=0.0 exactly reproduces C_Teknon_only
# alpha=1.0 exactly reproduces F_all with class balancing

phaseA_rows = []
for alpha in ALPHA_GRID:
    sample_weights = build_teknon_anchored_weights(
        dataset_train, y_train_pool, alpha=alpha, class_balance=True
    )

    folds = cv_double_metric_weighted(   # see below
        REF_MODEL, X_train_binned, y_train_pool, IS_TEKNON,
        groups=PATIENT_GROUPS,
        sample_weights=sample_weights,
    )
    tag = f"alpha={alpha:.2f}"
    phaseA_rows.append(summarise(folds, tag))
    print(f"  {tag} | Teknon bal_acc = {folds['t_balanced_accuracy'].mean():.3f} ± {folds['t_balanced_accuracy'].std():.3f}")

phaseA = (
    pd.DataFrame(phaseA_rows)
    .sort_values("t_balanced_accuracy_mean", ascending=False)
    .reset_index(drop=True)
)
phaseA.to_csv("task2_phaseA_dataset_selection.csv", index=False)
print(f"\n  [Phase A completed in {time.time() - t0:.0f}s]")
print("\n  Top results (sorted by Teknon balanced accuracy):")
display_cols = [
    "config",
    "t_balanced_accuracy_mean", "t_balanced_accuracy_std",
    "g_balanced_accuracy_mean",
    "t_macro_f1_mean",
    "t_roc_auc_mean",
]
print(phaseA[display_cols].round(3).to_string(index=False))

# ---- Freeze winning dataset combination ----
# best_dataset_combo will be something like "alpha=0.20"
best_dataset_combo = phaseA.iloc[0]["config"]

print(f"\n[WINNER PHASE A] Locked in optimal configuration: '{best_dataset_combo}'")

# Parse the winning float value out of the string (e.g., "alpha=0.20" -> 0.20)
try:
    BEST_ALPHA = float(best_dataset_combo.split("=")[1])
except:
    BEST_ALPHA = 0.00  # Safe fallback if string parsing glitches

# Since Claude's new approach dynamically weights the full pool, 
# we train on ALL datasets (F_all) and let the optimal weights handle domain shift!
BEST_DATASETS     = ["Sims", "CARTO", "Database", "Teknon"]
BEST_DATASET_MASK = np.isin(dataset_train, BEST_DATASETS)

# Recompute the optimal anchor weights using the winning alpha to carry into Phase B & C
FINAL_SAMPLE_WEIGHTS = build_teknon_anchored_weights(
    dataset_train[BEST_DATASET_MASK], 
    y_train_pool[BEST_DATASET_MASK], 
    alpha=BEST_ALPHA, 
    class_balance=True
)
print(f"-> Generated optimal sample weights using alpha={BEST_ALPHA:.2f} for downstream phases.")




# ==============================================================================
# 4.  PHASE B — LEAD SELECTION
# ==============================================================================

print("\n" + "=" * 70)
print("PHASE B — Lead Selection")
print("=" * 70)

LEAD_SUBSETS = {
    "all_12":           LEAD_NAMES,
    "precordial_V1_V6": ["V1", "V2", "V3", "V4", "V5", "V6"],
    "V1_V4":            ["V1", "V2", "V3", "V4"],
    "V2_V3_V4":         ["V2", "V3", "V4"],
    "V2_V3":            ["V2", "V3"],
    "V2_only":          ["V2"],
    "limb_only":        ["I", "II", "III", "aVR", "aVL", "aVF"],
    "V1_V2_V3_aVL":     ["V1", "V2", "V3", "aVL"],
}

def cols_for_leads(lead_list: list) -> list:
    cols = []
    for ld in lead_list:
        cols.extend(LEAD_TO_COLS[ld])
    return sorted(cols)

phaseB_rows = []
t0 = time.time()
for tag, leads in LEAD_SUBSETS.items():
    cols  = cols_for_leads(leads)
    folds = cv_double_metric(
        REF_MODEL, X_train_binned, y_train_pool, IS_TEKNON,
        groups=PATIENT_GROUPS, dataset_train_mask=BEST_DATASET_MASK, lead_cols=cols,
    )
    row = summarise(folds, tag)
    phaseB_rows.append(row)
    print(f"  {tag:22s} | Teknon bal_acc = {folds['t_balanced_accuracy'].mean():.3f}")

phaseB = pd.DataFrame(phaseB_rows).sort_values("t_balanced_accuracy_mean", ascending=False).reset_index(drop=True)
phaseB.to_csv("task2_phaseB_lead_selection.csv", index=False)

best_lead_combo = phaseB.iloc[0]["config"]
BEST_LEADS      = LEAD_SUBSETS[best_lead_combo]
BEST_COLS       = cols_for_leads(BEST_LEADS)


# ==============================================================================
# 5.  PHASE C — MODEL SELECTION
# ==============================================================================

print("\n" + "=" * 70)
print("PHASE C — Model Selection")
print("=" * 70)

def build_models(random_state: int = RANDOM_STATE) -> dict:
    models = {}

    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="ovr", solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=random_state,
        )),
    ])

    models["SVM_OvR"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            estimator=OneVsRestClassifier(
                LinearSVC(class_weight="balanced", max_iter=5000, random_state=random_state)
            ),
            cv=3,
        )),
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=random_state, n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", random_state=random_state, n_jobs=-1,
    )

    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu", max_iter=3000,
            early_stopping=True, validation_fraction=0.1, random_state=random_state,
        )),
    ])

    if XGB_AVAILABLE:
        # scale_pos_weight is omitted; training loops compute and supply sample_weights dynamically
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=N_CLASSES,
            eval_metric="mlogloss", random_state=random_state,
            n_jobs=-1, use_label_encoder=False,
        )

    return models


phaseC_rows = []
t0 = time.time()
for name, m in build_models().items():
    folds = cv_double_metric(
        m, X_train_binned, y_train_pool, IS_TEKNON,
        groups=PATIENT_GROUPS, dataset_train_mask=BEST_DATASET_MASK, lead_cols=BEST_COLS,
    )
    phaseC_rows.append(summarise(folds, name))
    print(f"  {name:20s} | Teknon bal_acc = {folds['t_balanced_accuracy'].mean():.3f} ± {folds['t_balanced_accuracy'].std():.3f}")

phaseC = pd.DataFrame(phaseC_rows).sort_values("t_balanced_accuracy_mean", ascending=False).reset_index(drop=True)
phaseC.to_csv("task2_phaseC_model_selection.csv", index=False)

best_model_name = phaseC.iloc[0]["config"]


# ==============================================================================
# 6.  FINAL MODEL — TRAIN ONCE, EVALUATE ONCE ON TEKNON TEST
# ==============================================================================

print("\n" + "=" * 70)
print("§6  Final Model — Single Train → Single Test Evaluation")
print("=" * 70)

X_final_train = X_train_binned[BEST_DATASET_MASK][:, BEST_COLS]
y_final_train = y_train_pool[BEST_DATASET_MASK]
X_final_test  = X_test_binned[:, BEST_COLS]

# Re-instantiate pristine unfitted chosen architecture
final_model = build_models()[best_model_name]

# Apply sample weighting rules onto the final un-blinded model configuration
base_clf = final_model.steps[-1][1] if isinstance(final_model, Pipeline) else final_model
if XGB_AVAILABLE and isinstance(base_clf, XGBClassifier):
    final_sw = compute_sample_weight(class_weight="balanced", y=y_final_train)
    if isinstance(final_model, Pipeline):
        fit_params = {f"{final_model.steps[-1][0]}__sample_weight": final_sw}
        final_model.fit(X_final_train, y_final_train, **fit_params)
    else:
        final_model.fit(X_final_train, y_final_train, sample_weight=final_sw)
else:
    final_model.fit(X_final_train, y_final_train)

y_pred  = final_model.predict(X_final_test)
y_score = _scores_from(final_model, X_final_test)

final_metrics = metrics_block(y_test, y_pred, y_score)
final_metrics["weighted_f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)

pd.DataFrame([{"model": best_model_name, **final_metrics}]).to_csv("task2_final_test_results.csv", index=False)

print("\n  === FINAL Teknon Test Results ===")
for k, v in final_metrics.items():
    print(f"    {k:24s}: {v:.4f}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

# ==============================================================================
# 6.1 FIGURE EXPORT: COMBINED CONFUSION MATRIX & MULTI-CLASS ROC CURVES
# ==============================================================================
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

# Setup a clean 1x2 panel layout matching fig_exp1_final_confusion_roc.png
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel A: Confusion Matrix Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=axes[0], cmap="Blues", colorbar=True, values_format="d")
axes[0].set_title("Confusion Matrix (Held-out Teknon Test)", fontsize=11, fontweight="bold", pad=10)

# Panel B: Multi-class One-vs-Rest (OvR) ROC curves
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i in range(N_CLASSES):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
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
axes[1].set_title("One-vs-Rest ROC Curves (Teknon Cohort)", fontsize=11, fontweight="bold", pad=10)
axes[1].legend(loc="lower right", fontsize=9, frameon=True)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_exp1_final_confusion_roc.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  [Saved] fig_exp1_final_confusion_roc.png (Combined Evaluation Asset)")

# ---- §6.2  Bootstrap CI ----
rng  = np.random.RandomState(RANDOM_STATE)
boot = []
n_t  = len(y_test)
for _ in range(2000):
    idx = rng.randint(0, n_t, n_t)
    if len(np.unique(y_test[idx])) < N_CLASSES:
        continue
    boot.append(balanced_accuracy_score(y_test[idx], y_pred[idx]))
boot = np.array(boot)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"    Balanced accuracy: {final_metrics['balanced_accuracy']:.3f} (95% bootstrap CI: {lo:.3f} – {hi:.3f})")


# ==============================================================================
# 7.  INTERPRETABILITY & FEATURE IMPORTANCE RANKINGS
# ==============================================================================
print("\n" + "=" * 70)
print("§7  Interpretability")
print("=" * 70)

SELECTED_FEATURE_NAMES = [FEATURE_NAMES[c] for c in BEST_COLS]

def get_importance(model, X: np.ndarray, y: np.ndarray, names: list, random_state: int = RANDOM_STATE) -> tuple:
    base = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if isinstance(base, CalibratedClassifierCV):
        base = base.estimator
    if isinstance(base, OneVsRestClassifier):
        base = base.estimator

    if hasattr(base, "feature_importances_"):
        return np.asarray(base.feature_importances_), "native"

    r = permutation_importance(model, X, y, n_repeats=20, random_state=random_state, n_jobs=-1, scoring="balanced_accuracy")
    return np.clip(r.importances_mean, 0, None), "permutation"

importance, imp_kind = get_importance(final_model, X_final_test, y_test, SELECTED_FEATURE_NAMES)

# ---- Plot Horizontal Feature Importance Bars (fig_exp1_importance.png) ----
sort_idx = np.argsort(importance)
# Limit to top 15 features for presentation readability if feature set is massive
top_indices = sort_idx[-15:] if len(sort_idx) > 15 else sort_idx

fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
ax_imp.barh(range(len(top_indices)), importance[top_indices], color="#4682B4", edgecolor="none", height=0.6)
ax_imp.set_yticks(range(len(top_indices)))
ax_imp.set_yticklabels([SELECTED_FEATURE_NAMES[i] for i in top_indices], fontsize=9)
ax_imp.set_xlabel(f"Importance Score ({imp_kind} metric weights)", fontsize=10)
ax_imp.set_title("Top Selected Feature Importance Ranking Map", fontsize=11, fontweight="bold", pad=12)
ax_imp.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("fig_exp1_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig_imp)
print("  [Saved] fig_exp1_importance.png (Feature Priority Map)")


# ==============================================================================
# 7.1 SHAP BEESWARM MULTI-PANEL GENERATOR
# ==============================================================================
base_for_shap = final_model.steps[-1][1] if isinstance(final_model, Pipeline) else final_model
if isinstance(base_for_shap, CalibratedClassifierCV):
    base_for_shap = base_for_shap.estimator
if isinstance(base_for_shap, OneVsRestClassifier):
    base_for_shap = base_for_shap.estimator

if SHAP_AVAILABLE and hasattr(base_for_shap, "feature_importances_"):
    try:
        explainer = shap.TreeExplainer(base_for_shap)
        shap_values = explainer.shap_values(X_final_test)

        if isinstance(shap_values, list) and len(shap_values) == N_CLASSES:
            fig_shap, axes_shap = plt.subplots(1, N_CLASSES, figsize=(6 * N_CLASSES, 4.5))
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                plt.sca(axes_shap[cls_idx])
                shap.summary_plot(
                    shap_values[cls_idx], X_final_test, 
                    feature_names=SELECTED_FEATURE_NAMES, 
                    show=False, plot_type="dot"
                )
                axes_shap[cls_idx].set_title(f"SHAP Attribution — {cls_name}", fontsize=11, fontweight="bold")
            
            plt.tight_layout()
            plt.savefig("fig_exp1_shap_beeswarm.png", dpi=150, bbox_inches="tight")
            plt.close(fig_shap)
            print("  [Saved] fig_exp1_shap_beeswarm.png (Multi-class Localized Attributions)")
    except Exception as exc:
        print(f"  [WARN] SHAP processing halted: {exc!s}")

print("\n" + "=" * 70)
print("§8  Summary — Execution Complete Safely. All figures locked.")
print("=" * 70)