"""
Stage 2 — Extract AE features, train Random Forest, evaluate.

Evaluation notes
----------------
Cell-level AUC: each cell is NOT an independent observation. Cells from the
same patient share staining batch, slide preparation, and microscope settings.
The effective sample count equals the number of patients, not cells. Cell AUC
is reported for completeness but should not be the primary metric.

Patient-level AUC: computed by averaging per-cell probabilities to one score
per patient, then ranking. With only 4 validation patients (2-vs-2), the AUC
can only take three values: 0.0, 0.5, or 1.0. A random model achieves AUC=1.0
with probability 1/3. This metric has near-zero statistical power and must be
interpreted alongside the leakage diagnostic below.

Leakage diagnostic: required check — if AE features predict patient ID well,
the RF is likely exploiting per-patient slide/staining artefacts rather than
cancer biology. See check_patient_id_leakage() below.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import joblib
from tqdm import tqdm
from ae_models import ResidualAE

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths / splits ────────────────────────────────────────────────────────────
SUB_ROOT     = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(SUB_ROOT)
VAL_PATIENTS = ["pat_07", "pat_10", "pat_05", "pat_16"]


class CellDataset(Dataset):
    def __init__(self, df, split="train"):
        self.df    = df.reset_index(drop=True)
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        filename = row["Name"]
        bf = Image.open(os.path.join(ROOT, "BF", self.split, filename)).convert("RGB")
        fl = Image.open(os.path.join(ROOT, "FL", self.split, filename)).convert("RGB")
        img = torch.cat([TF.to_tensor(bf), TF.to_tensor(fl)], dim=0)
        imin, imax = img.min(), img.max()
        img = (img - imin) / (imax - imin + 1e-6)
        label = torch.tensor(float(row["Diagnosis"]), dtype=torch.float32)
        return img, label


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_features(model, device, loader, name=""):
    """Return (features [N×256], labels [N])."""
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f"  features [{name}]", leave=False):
            _, feat = model(imgs.to(device))
            feats.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def safe_auc(y_true, y_score, name=""):
    """roc_auc_score with a guard for single-class folds."""
    if len(np.unique(y_true)) < 2:
        print(f"  [skip] {name}: only one class in y_true — AUC undefined")
        return None
    return float(roc_auc_score(y_true, y_score))


def _unique_label(series):
    """Aggregation: assert all cells of a patient carry the same label."""
    vals = series.unique()
    if len(vals) != 1:
        raise ValueError(f"Inconsistent patient labels found: {vals.tolist()}")
    return float(vals[0])


def check_patient_id_leakage(X_train, train_patient_ids):
    """
    5-fold CV: can a classifier predict patient ID from AE features?
    Uses only training patients so val evaluation is unaffected.

    High accuracy relative to chance (1/n_patients) means features encode
    per-patient imaging artefacts — the RF may be exploiting batch effects,
    not cancer biology.
    """
    le      = LabelEncoder()
    y_pid   = le.fit_transform(train_patient_ids)
    n       = len(le.classes_)
    chance  = 1.0 / n

    clf     = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    acc     = cross_val_score(clf, X_train, y_pid, cv=5, scoring="accuracy").mean()
    ratio   = acc / chance

    print("\n" + "=" * 60)
    print("[Leakage Diagnostic] Patient-ID prediction from AE features")
    print(f"  Train patients   : {n}")
    print(f"  Chance level     : {chance:.3f}  (1/{n})")
    print(f"  5-fold CV acc    : {acc:.3f}  ({ratio:.1f}x chance)")
    if ratio > 3.0:
        print("  *** CRITICAL: Features strongly encode patient identity.")
        print("  *** RF results likely reflect slide/staining memorisation,")
        print("  *** not cancer biology. Results should not be trusted as-is.")
    elif ratio > 2.0:
        print("  !! WARNING: Moderate patient-identity signal detected.")
        print("  !! Interpret RF metrics cautiously.")
    else:
        print("  OK: No strong patient-identity signal detected.")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def train_rf_and_evaluate():
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else "cpu"
    )
    print(f"[helper] device = {device}")

    # Load AE
    ae = ResidualAE().to(device)
    model_path = os.path.join(SUB_ROOT, "best_residual_ae.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained AE at {model_path}. Run train.py first.")
    ae.load_state_dict(torch.load(model_path, map_location=device))
    ae.eval()
    print(f"[helper] Loaded AE from {model_path}")

    # Build splits
    df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    df["Patient"] = df["Name"].apply(lambda x: "_".join(x.split("_")[:2]))

    # Validate that every VAL_PATIENT is present in the CSV
    found = df["Patient"].unique()
    missing = [p for p in VAL_PATIENTS if p not in found]
    if missing:
        raise ValueError(
            f"VAL_PATIENTS not found in train.csv: {missing}. "
            f"Check filename format (expected 'pat_XX_...'). Got: {sorted(found)}"
        )

    # RF trains on ALL non-val patients (AE_STOP_PATIENT included — it was only
    # withheld from AE *training*, not from label-dependent decisions).
    train_df = df[~df["Patient"].isin(VAL_PATIENTS)].reset_index(drop=True)
    val_df   = df[ df["Patient"].isin(VAL_PATIENTS)].reset_index(drop=True)

    print(f"[helper] RF train patients : {sorted(train_df['Patient'].unique())}")
    print(f"[helper] RF val   patients : {sorted(val_df['Patient'].unique())}")

    train_loader = DataLoader(CellDataset(train_df), batch_size=128, shuffle=False, num_workers=4)
    val_loader   = DataLoader(CellDataset(val_df),   batch_size=128, shuffle=False, num_workers=4)

    # Extract features (shuffle=False so row order matches df order)
    print("\n[helper] Extracting AE features…")
    X_train, y_train = extract_features(ae, device, train_loader, "train")
    X_val,   y_val   = extract_features(ae, device, val_loader,   "val")

    # ── Leakage diagnostic ────────────────────────────────────────────────────
    check_patient_id_leakage(X_train, train_df["Patient"].values)

    # ── Train RF ──────────────────────────────────────────────────────────────
    print("[helper] Training Random Forest…")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=4,
        random_state=SEED, n_jobs=-1,
        oob_score=True,   # unbiased training-set proxy (replaces misleading in-sample AUC)
    )
    rf.fit(X_train, y_train)

    rf_path = os.path.join(SUB_ROOT, "best_rf_model.joblib")
    joblib.dump(rf, rf_path)
    print(f"[helper] Saved RF to {rf_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_probs = rf.predict_proba(X_val)[:, 1]

    # OOB accuracy as a rough sanity-check on training behaviour.
    # Note: OOB reports accuracy, not AUC — use it only to spot catastrophic failure.
    print(f"\n[helper] RF OOB accuracy (unbiased, ≠ AUC): {rf.oob_score_:.4f}")

    # Cell-level AUC
    n_val_patients = val_df["Patient"].nunique()
    val_cell_auc   = safe_auc(y_val, val_probs, "val cell-level")

    print("\n" + "-" * 60)
    print("Cell-Level Results")
    print(f"  NOTE: {len(val_df)} cells from {n_val_patients} patients are NOT independent.")
    print(f"  Effective n ≈ {n_val_patients} (number of val patients), not cell count.")
    if val_cell_auc is not None:
        print(f"  Val Cell AUC : {val_cell_auc:.4f}  [treat as descriptive only]")

    # Patient-level AUC
    # val_probs aligns with val_df because DataLoader used shuffle=False and val_df
    # was reset_index(drop=True), so positional assignment is correct.
    val_df = val_df.copy()
    val_df["Prob"] = val_probs

    val_pat = (
        val_df
        .groupby("Patient")
        .agg(Prob=("Prob", "mean"), Diagnosis=("Diagnosis", _unique_label))
        .reset_index()
    )

    n_val      = len(val_pat)
    n_possible = n_val * (n_val - 1) // 2 + 1
    val_pat_auc = safe_auc(val_pat["Diagnosis"].values, val_pat["Prob"].values, "val patient-level")

    print(f"\nPatient-Level Results  ({n_val} patients)")
    print(f"  STATISTICAL CAUTION:")
    print(f"  AUC over {n_val} patients has only {n_possible} possible distinct values.")
    if n_val <= 4:
        print(f"  With a 2-vs-2 split, AUC ∈ {{0.0, 0.5, 1.0}}.")
        print(f"  A random model achieves AUC=1.0 with prob ~1/3.")
        print(f"  This result is NOT statistically meaningful on its own.")
    if val_pat_auc is not None:
        print(f"  Val Patient AUC : {val_pat_auc:.4f}")

    print("\n  Per-patient breakdown:")
    for _, row in val_pat.iterrows():
        label = "cancer" if int(row["Diagnosis"]) == 1 else "healthy"
        print(f"    {row['Patient']}  true={label}  pred_mean={row['Prob']:.4f}")
    print("-" * 60)

    return rf


if __name__ == "__main__":
    train_rf_and_evaluate()
