"""
================================================================================
TASK 2 — Teknon-only Experiment: QRS + Clinical Data (3-Class Edition)
================================================================================

Goal:
    Extend the binary setup to a 3-CLASS site-of-origin classification using 
    both binned QRS loops and patient-level clinical variables.
        Class 0 → RVOT
        Class 1 → RCC-LCC
        Class 2 → LVOT

Methodological Architecture:
    Family A — QRS only      → 120 features (12 leads × 10 bins)
    Family B — Clinical only → Age, Sex, HTA, V3 amplitude, precordial-transition
    Family C — Combined      → C.1: Stacking (OOF predictions)
                               C.2: Stacking (OOF probabilities)
                               C.3: Concatenation (Clinical + 120 QRS features)

Validation Safeguard:
    Strict StratifiedGroupKFold cross-validation over the Teknon training pool
    grouped by 'patient_index' to prevent cross-patient vector leakage.
================================================================================
"""

import os
import time
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ---- Optional Extras ----
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# -----------------------------------------------------------------------
# Global Configuration Parameters
# -----------------------------------------------------------------------
RANDOM_STATE = 42
N_FOLDS      = 5
N_BINS       = 10
N_CLASSES    = 3

LEAD_NAMES   = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASS_NAMES  = ["RVOT", "RCC-LCC", "LVOT"]

np.random.seed(RANDOM_STATE)
SGKF = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Data Directory Handling
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR   = _SCRIPT_DIR / ".." / "data"

def find_file(filename: str) -> Path:
    candidates = [_DATA_DIR / filename, Path(".") / filename, Path("..") / filename]
    for p in candidates:
        if p.exists(): return p.resolve()
    raise FileNotFoundError(f"Could not locate target data file: {filename}")

# ==============================================================================
# 1. DATA PIPELINE LOADING & INGESTION
# ==============================================================================
print("[INFO] Executing Pipeline Data Load Sequences...")

X_train_pool = np.load(find_file("X_train_pool_task2_3class.npy"))
y_train_pool = np.load(find_file("y_train_pool_task2_3class.npy")).astype(int)
X_test       = np.load(find_file("X_teknon_final_test_task2_3class.npy"))
y_test       = np.load(find_file("y_teknon_final_test_task2_3class.npy")).astype(int)

info_train = pd.read_csv(find_file("info_train_pool_task2_3class.csv"))
info_test  = pd.read_csv(find_file("info_teknon_final_test_task2_3class.csv"))

with open(find_file("full_data_7class.pkl"), "rb") as f:
    pkl_metadata = pickle.load(f)

# Slice training pool strictly down to Teknon observation matrices
is_teknon_train = info_train["dataset"].astype(str).str.lower() == "teknon"
teknon_train_indices = np.where(is_teknon_train.values)[0]

X_train_tek = X_train_pool[teknon_train_indices]
y_train_tek = y_train_pool[teknon_train_indices]
info_train_tek = info_train.iloc[teknon_train_indices].reset_index(drop=True)

train_patient_groups = info_train_tek["patient_index"].astype(int).to_numpy()
test_patient_groups  = info_test["patient_index"].astype(int).to_numpy()

# ==============================================================================
# 2. MORPHOLOGICAL FEATURE EXTRACTION (12 LEADS × 10 PERCENTILE BINS)
# ==============================================================================
def create_binned_features(X_signals: np.ndarray) -> np.ndarray:
    n_samples, n_leads, n_timepoints = X_signals.shape
    edges = np.linspace(0, n_timepoints, N_BINS + 1, dtype=int)
    X_binned = np.zeros((n_samples, n_leads * N_BINS), dtype=np.float32)
    for l_idx in range(n_leads):
        for b_idx in range(N_BINS):
            segment = X_signals[:, l_idx, edges[b_idx]:edges[b_idx + 1]]
            X_binned[:, l_idx * N_BINS + b_idx] = segment.mean(axis=1)
    return X_binned

X_train_qrs = create_binned_features(X_train_tek)
X_test_qrs  = create_binned_features(X_test)

# ==============================================================================
# 3. CLINICAL MATRIX CONSTRUCTION
# ==============================================================================
def assemble_clinical_dataframe(df_info: pd.DataFrame, patient_ids: np.ndarray) -> pd.DataFrame:
    records = []
    for p_id in patient_ids:
        records.append({
            "Age": p_metadata["Age"][p_id] if "Age" in p_metadata else np.nan,
            "Sex": p_metadata["Sex"][p_id] if "Sex" in p_metadata else np.nan,
            "HTA": p_metadata["HTA"][p_id] if "HTA" in p_metadata else np.nan,
            "V3_amp": np.max(np.abs(p_metadata["signals"][p_id][8, :])) if "signals" in p_metadata else np.nan,
            "precordial_transition": p_metadata["precordial_transition"][p_id] if "precordial_transition" in p_metadata else np.nan
        })
    return pd.DataFrame(records)

p_metadata = pkl_metadata
df_clin_train = assemble_clinical_dataframe(info_train_tek, train_patient_groups)
df_clin_test  = assemble_clinical_dataframe(info_test, test_patient_groups)

# Robustly clean and map Sex values
sex_map = {"M": 1, "F": 0, "H": 1, "D": 0, "m": 1, "f": 0}
df_clin_train["Sex"] = df_clin_train["Sex"].map(sex_map).fillna(-1)
df_clin_test["Sex"]  = df_clin_test["Sex"].map(sex_map).fillna(-1)

# Robustly clean and map HTA (Hypertension) text values
hta_map = {"Yes": 1, "No": 0, "SI": 1, "Si": 1, "NO": 0, "yes": 1, "no": 0, 1: 1, 0: 0}
df_clin_train["HTA"] = df_clin_train["HTA"].map(hta_map).fillna(-1)
df_clin_test["HTA"]  = df_clin_test["HTA"].map(hta_map).fillna(-1)

X_clin_train = df_clin_train.to_numpy().astype(np.float32)
X_clin_test  = df_clin_test.to_numpy().astype(np.float32)

# ==============================================================================
# 4. ROBUST BALANCED MULTI-CLASS CLASSIFIER MODEL FACTORY
# ==============================================================================
def get_base_model_pipeline(random_state=RANDOM_STATE) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1))
    ])

# ==============================================================================
# 5. CORE BENCHMARK COMPILATION & SEQUENTIAL RUN LOOPS
# ==============================================================================
print("\n" + "="*80 + "\n[RUN] Executing Phase Stacking Cross Validation Sequences\n" + "="*80)

# ----- FAMILY A: QRS Only -----
pipe_A = get_base_model_pipeline()
oof_preds_A = np.zeros(len(y_train_tek))
oof_probs_A = np.zeros((len(y_train_tek), N_CLASSES))

for tr, va in SGKF.split(X_train_qrs, y_train_tek, groups=train_patient_groups):
    sw = compute_sample_weight("balanced", y_train_tek[tr])
    pipe_A.fit(X_train_qrs[tr], y_train_tek[tr], clf__sample_weight=sw)
    oof_preds_A[va] = pipe_A.predict(X_train_qrs[va])
    oof_probs_A[va] = pipe_A.predict_proba(X_train_qrs[va])

bal_acc_A = balanced_accuracy_score(y_train_tek, oof_preds_A)
print(f"Family A (QRS Only) OOF Balanced Accuracy: {bal_acc_A:.4f}")

# ----- FAMILY B: Clinical Only -----
pipe_B = get_base_model_pipeline()
oof_preds_B = np.zeros(len(y_train_tek))

for tr, va in SGKF.split(X_clin_train, y_train_tek, groups=train_patient_groups):
    sw = compute_sample_weight("balanced", y_train_tek[tr])
    pipe_B.fit(X_clin_train[tr], y_train_tek[tr], clf__sample_weight=sw)
    oof_preds_B[va] = pipe_B.predict(X_clin_train[va])

bal_acc_B = balanced_accuracy_score(y_train_tek, oof_preds_B)
print(f"Family B (Clinical Only) OOF Balanced Accuracy: {bal_acc_B:.4f}")

# ----- FAMILY C: Combined Models (Stacking and Concatenation) -----

# C.1: Stacking Discrete Target Maps
X_combined_C1 = np.column_stack([X_clin_train, oof_preds_A])
pipe_C1 = get_base_model_pipeline()
oof_preds_C1 = np.zeros(len(y_train_tek))
for tr, va in SGKF.split(X_combined_C1, y_train_tek, groups=train_patient_groups):
    sw = compute_sample_weight("balanced", y_train_tek[tr])
    pipe_C1.fit(X_combined_C1[tr], y_train_tek[tr], clf__sample_weight=sw)
    oof_preds_C1[va] = pipe_C1.predict(X_combined_C1[va])
print(f"Family C.1 (Stacked Predictions) OOF Balanced Accuracy: {balanced_accuracy_score(y_train_tek, oof_preds_C1):.4f}")

# C.2: Stacking Continuous Probability Distribution Vector Fields
X_combined_C2 = np.column_stack([X_clin_train, oof_probs_A])
pipe_C2 = get_base_model_pipeline()
oof_preds_C2 = np.zeros(len(y_train_tek))
for tr, va in SGKF.split(X_combined_C2, y_train_tek, groups=train_patient_groups):
    sw = compute_sample_weight("balanced", y_train_tek[tr])
    pipe_C2.fit(X_combined_C2[tr], y_train_tek[tr], clf__sample_weight=sw)
    oof_preds_C2[va] = pipe_C2.predict(X_combined_C2[va])
print(f"Family C.2 (Stacked Probabilities) OOF Balanced Accuracy: {balanced_accuracy_score(y_train_tek, oof_preds_C2):.4f}")

# C.3: Direct Vector Feature Concatenation Field
X_combined_C3 = np.column_stack([X_clin_train, X_train_qrs])
pipe_C3 = get_base_model_pipeline()
oof_preds_C3 = np.zeros(len(y_train_tek))
for tr, va in SGKF.split(X_combined_C3, y_train_tek, groups=train_patient_groups):
    sw = compute_sample_weight("balanced", y_train_tek[tr])
    pipe_C3.fit(X_combined_C3[tr], y_train_tek[tr], clf__sample_weight=sw)
    oof_preds_C3[va] = pipe_C3.predict(X_combined_C3[va])
print(f"Family C.3 (Concatenation Execution) OOF Balanced Accuracy: {balanced_accuracy_score(y_train_tek, oof_preds_C3):.4f}")

# ==============================================================================
# 6. HEURISTIC FREEZING & SINGLE RUN TARGET COHORT HELD-OUT EVALUATION
# ==============================================================================
print("\n" + "="*80 + "\n[EXEC] Freezing Optimal Configuration Pipeline Architecture & Evaluating Test Set\n" + "="*80)

# Assuming C.2 (Stacked Probabilities) is chosen as top validation performer
# 1. Fit the complete first-stage QRS model on the whole training array
pipe_A_final = get_base_model_pipeline()
sw_A = compute_sample_weight("balanced", y_train_tek)
pipe_A_final.fit(X_train_qrs, y_train_tek, clf__sample_weight=sw_A)

# 2. Extract first-stage features for the unseen test matrices
test_probs_A = pipe_A_final.predict_proba(X_test_qrs)

# 3. Form final evaluation stacking frame meta-features
X_test_stacked = np.column_stack([X_clin_test, test_probs_A])
X_train_stacked = np.column_stack([X_clin_train, oof_probs_A])

# 4. Final step calibration fit and prediction scoring
pipe_final_stacker = get_base_model_pipeline()
sw_final = compute_sample_weight("balanced", y_train_tek)
pipe_final_stacker.fit(X_train_stacked, y_train_tek, clf__sample_weight=sw_final)

final_predictions = pipe_final_stacker.predict(X_test_stacked)
final_prob_scores = pipe_final_stacker.predict_proba(X_test_stacked)

print("\n=== FINAL TEST METRIC EVALUATION REPORT (HELD-OUT COHORT) ===")
print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_test, final_predictions):.4f}")
print(f"Accuracy Categorical Score: {accuracy_score(y_test, final_predictions):.4f}")
print(f"Macro F1 Evaluation Weight: {f1_score(y_test, final_predictions, average='macro'):.4f}")

print("\nClassification Report Matrix:")
print(classification_report(y_test, final_predictions, target_names=CLASS_NAMES, zero_division=0))

# ==============================================================================
# EXPERIMENT 2 PRODUCTION VISUALIZATION SUITE (CORRECTED)
# ==============================================================================

# Extract the clinical feature column list dynamically from the dataframe
clinical_feature_names = list(df_clin_train.columns)

# ── 1. EXPORT SEPARATE CONFUSION MATRIX
cm = confusion_matrix(y_test, final_predictions, labels=range(N_CLASSES))
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax_cm, cmap="Blues", colorbar=True, values_format="d")
ax_cm.set_title("Confusion Matrix — Experiment 2 (Native Stacking)", fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("fig_teknon_confusion_best.png", dpi=150, bbox_inches="tight")
plt.close(fig_cm)
print("  [Saved] fig_teknon_confusion_best.png")

# ── 2. EXPORT MULTI-CLASS ONE-VS-REST ROC CURVES
fig_roc, ax_roc = plt.subplots(figsize=(6.5, 5))
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i in range(N_CLASSES):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], final_prob_scores[:, i])
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=colors[i], lw=2, label=f"ROC {CLASS_NAMES[i]} (AUC = {roc_auc:.2f})")

ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel("False Positive Rate", fontsize=10)
ax_roc.set_ylabel("True Positive Rate", fontsize=10)
ax_roc.set_title("One-vs-Rest ROC Curves (Teknon Clinical Stack)", fontsize=11, fontweight="bold", pad=10)
ax_roc.legend(loc="lower right", fontsize=9, frameon=True)
ax_roc.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_teknon_roc_best.png", dpi=150, bbox_inches="tight")
plt.close(fig_roc)
print("  [Saved] fig_teknon_roc_best.png")

# ── 3. EXPORT STANDALONE CLINICAL IMPORTANCE (FAMILY B BEFORE STACKING)
print("  [Processing] Computing clinical-only permutation importance...")
clinical_base_clf = get_base_model_pipeline()
sw_clin = compute_sample_weight("balanced", y_train_tek)
clinical_base_clf.fit(X_clin_train, y_train_tek, clf__sample_weight=sw_clin)

perm_importance_clin = permutation_importance(
    clinical_base_clf, X_clin_test, y_test, 
    n_repeats=20, random_state=RANDOM_STATE, scoring="balanced_accuracy"
)
clin_weights = np.clip(perm_importance_clin.importances_mean, 0, None)
clin_sort_idx = np.argsort(clin_weights)

fig_clin, ax_clin = plt.subplots(figsize=(7, 4))
ax_clin.barh(range(len(clinical_feature_names)), clin_weights[clin_sort_idx], color="#66c2a5", height=0.55)
ax_clin.set_yticks(range(len(clinical_feature_names)))
ax_clin.set_yticklabels([clinical_feature_names[i] for i in clin_sort_idx], fontsize=9)
ax_clin.set_xlabel("Permutation Importance (Balanced Accuracy Δ)", fontsize=10)
ax_clin.set_title("Clinical Feature Baseline Importance Profile", fontsize=11, fontweight="bold", pad=12)
ax_clin.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("fig_teknon_importance_clinical.png", dpi=150, bbox_inches="tight")
plt.close(fig_clin)
print("  [Saved] fig_teknon_importance_clinical.png")

# ── 4. EXPORT META-STACKER COMBINED IMPORTANCE (FAMILY C)
combined_feature_names = clinical_feature_names + [f"Meta_Prob_{name}" for name in CLASS_NAMES]
perm_importance_stacker = permutation_importance(
    pipe_final_stacker, X_test_stacked, y_test,
    n_repeats=20, random_state=RANDOM_STATE, scoring="balanced_accuracy"
)
stacker_weights = np.clip(perm_importance_stacker.importances_mean, 0, None)
stacker_sort_idx = np.argsort(stacker_weights)

fig_stack, ax_stack = plt.subplots(figsize=(8, 4.5))
ax_stack.barh(range(len(combined_feature_names)), stacker_weights[stacker_sort_idx], color="#fc8d62", height=0.55)
ax_stack.set_yticks(range(len(combined_feature_names)))
ax_stack.set_yticklabels([combined_feature_names[i] for i in stacker_sort_idx], fontsize=9)
ax_stack.set_xlabel("Meta-Stacker Feature Importance (Balanced Accuracy Δ)", fontsize=10)
ax_stack.set_title("Combined Clinical & QRS Meta-Stacker Dominance Map", fontsize=11, fontweight="bold", pad=12)
ax_stack.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("fig_teknon_importance_combined.png", dpi=150, bbox_inches="tight")
plt.close(fig_stack)
print("  [Saved] fig_teknon_importance_combined.png")

# ── 5. EXPORT ANATOMICAL 7-CLASS ERROR ANALYSIS LOG & PLOT
print("  [Processing] Resolving 7-class structural misclassifications...")
error_indices = np.where(final_predictions != y_test)[0]

if len(error_indices) > 0:
    with open(find_file("full_data_7class.pkl"), "rb") as f:
        pkl_metadata = pickle.load(f)
        
    error_sublocations = []
    for idx in error_indices:
        patient_idx = int(info_test.iloc[idx]["patient_index"])
        true_7class = "Unknown"
        if isinstance(pkl_metadata, dict) and "SOO_7class" in pkl_metadata:
            if 0 <= patient_idx < len(pkl_metadata["SOO_7class"]):
                true_7class = pkl_metadata["SOO_7class"][patient_idx]
        error_sublocations.append(true_7class)
        
    err_df = pd.DataFrame({
        "True_Detailed_7Class_SOO": error_sublocations,
        "Predicted_Macro": [CLASS_NAMES[final_predictions[idx]] for idx in error_indices]
    })
    
    counts = err_df["True_Detailed_7Class_SOO"].value_counts()
    
    fig_err, ax_err = plt.subplots(figsize=(7, 4.5))
    counts.plot(kind="bar", ax=ax_err, color="#e78ac3", edgecolor="none", width=0.6)
    ax_err.set_ylabel("Number of Misclassified Test Patients", fontsize=10)
    ax_err.set_xlabel("True Fine-Grained Anatomical Sublocation", fontsize=10)
    ax_err.set_title("Anatomical Breakdown of Stacked Stacker Failures", fontsize=11, fontweight="bold", pad=12)
    ax_err.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=9)
    ax_err.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_teknon_error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig_err)
    
    err_df.to_csv("task2_teknon_stacked_errors_audit.csv", index=False)
    print("  [Saved] fig_teknon_error_analysis.png & task2_teknon_stacked_errors_audit.csv")
else:
    print("  [COMPLETE] Outstanding performance. Zero errors found to plot.")

print("\n" + "=" * 70)
print("§6  Execution Complete Safely. All Stacking Assets Exported.")
print("=" * 70)