"""
ResNet-50 + Random Forest pipeline.

No training step needed — ResNet weights are frozen ImageNet weights.
Run this directly:  python helper_resnet.py
Then:               python inference_resnet.py
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import joblib
from tqdm import tqdm
from resnet_extractor import ResNetExtractor

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths / splits ────────────────────────────────────────────────────────────
SUB_ROOT     = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(SUB_ROOT)
VAL_PATIENTS = ["pat_07", "pat_10", "pat_05", "pat_16"]

# ImageNet normalisation stats — required for pretrained ResNet.
# Applied identically to BF (ch 0-2) and FL (ch 3-5).
_IMAGENET_MEAN = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
_normalize     = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


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
        img   = torch.cat([TF.to_tensor(bf), TF.to_tensor(fl)], dim=0)  # [0,1] per channel
        img   = _normalize(img)                                           # ImageNet normalisation
        label = torch.tensor(float(row["Diagnosis"]), dtype=torch.float32)
        return img, label


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_features(model, device, loader, name=""):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f"  features [{name}]", leave=False):
            feat = model(imgs.to(device))
            feats.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def safe_auc(y_true, y_score, name=""):
    if len(np.unique(y_true)) < 2:
        print(f"  [skip] {name}: only one class — AUC undefined")
        return None
    return float(roc_auc_score(y_true, y_score))


def _unique_label(series):
    vals = series.unique()
    if len(vals) != 1:
        raise ValueError(f"Inconsistent patient labels: {vals.tolist()}")
    return float(vals[0])


def check_patient_id_leakage(X_train, train_patient_ids):
    le    = LabelEncoder()
    y_pid = le.fit_transform(train_patient_ids)
    n     = len(le.classes_)
    chance = 1.0 / n
    clf   = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    acc   = cross_val_score(clf, X_train, y_pid, cv=5, scoring="accuracy").mean()
    ratio = acc / chance

    print("\n" + "=" * 60)
    print("[Leakage Diagnostic] Patient-ID prediction from ResNet features")
    print(f"  Train patients   : {n}")
    print(f"  Chance level     : {chance:.3f}  (1/{n})")
    print(f"  5-fold CV acc    : {acc:.3f}  ({ratio:.1f}x chance)")
    if ratio > 3.0:
        print("  *** CRITICAL: Features strongly encode patient identity.")
    elif ratio > 2.0:
        print("  !! WARNING: Moderate patient-identity signal. Interpret cautiously.")
    else:
        print("  OK: No strong patient-identity signal detected.")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def train_rf_and_evaluate():
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else "cpu"
    )
    print(f"[resnet] device = {device}")

    # Load frozen ResNet extractor (no checkpoint needed)
    model = ResNetExtractor().to(device)
    model.eval()
    print("[resnet] Loaded frozen ResNet-50 (ImageNet weights)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[resnet] Backbone params: {total_params:,}  (all frozen)")

    # Build splits
    df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    df["Patient"] = df["Name"].apply(lambda x: "_".join(x.split("_")[:2]))

    train_df = df[~df["Patient"].isin(VAL_PATIENTS)].reset_index(drop=True)
    val_df   = df[ df["Patient"].isin(VAL_PATIENTS)].reset_index(drop=True)

    print(f"[resnet] Train patients : {sorted(train_df['Patient'].unique())}")
    print(f"[resnet] Val   patients : {sorted(val_df['Patient'].unique())}")

    train_loader = DataLoader(CellDataset(train_df), batch_size=128,
                              shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(CellDataset(val_df),   batch_size=128,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Extract features
    print("\n[resnet] Extracting ResNet features…")
    X_train, y_train = extract_features(model, device, train_loader, "train")
    X_val,   y_val   = extract_features(model, device, val_loader,   "val")
    print(f"[resnet] Feature shape: {X_train.shape}  (4096-dim per cell)")

    # Leakage diagnostic
    check_patient_id_leakage(X_train, train_df["Patient"].values)

    # Train RF
    print("[resnet] Training Random Forest…")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=4,
        random_state=SEED, n_jobs=-1, oob_score=True,
    )
    rf.fit(X_train, y_train)
    print(f"[resnet] RF OOB accuracy (unbiased, ≠ AUC): {rf.oob_score_:.4f}")

    rf_path = os.path.join(SUB_ROOT, "best_rf_resnet.joblib")
    joblib.dump(rf, rf_path)
    print(f"[resnet] Saved RF to {rf_path}")

    # Evaluate
    val_probs      = rf.predict_proba(X_val)[:, 1]
    val_cell_auc   = safe_auc(y_val, val_probs, "val cell-level")
    n_val_patients = val_df["Patient"].nunique()

    print("\n" + "-" * 60)
    print("Cell-Level Results")
    print(f"  NOTE: cells from the same patient are NOT independent.")
    print(f"  Effective n ≈ {n_val_patients} (val patients), not cell count.")
    if val_cell_auc is not None:
        print(f"  Val Cell AUC : {val_cell_auc:.4f}  [descriptive only]")

    # Patient-level
    val_df = val_df.copy()
    val_df["Prob"] = val_probs

    val_pat = (
        val_df.groupby("Patient")
        .agg(Prob=("Prob", "mean"), Diagnosis=("Diagnosis", _unique_label))
        .reset_index()
    )

    n_val       = len(val_pat)
    val_pat_auc = safe_auc(val_pat["Diagnosis"].values, val_pat["Prob"].values,
                           "val patient-level")

    print(f"\nPatient-Level Results  ({n_val} patients)")
    print(f"  CAUTION: AUC over {n_val} patients has only "
          f"{n_val*(n_val-1)//2 + 1} possible values — near-zero statistical power.")
    if val_pat_auc is not None:
        print(f"  Val Patient AUC : {val_pat_auc:.4f}")

    print("\n  Per-patient breakdown:")
    for _, row in val_pat.iterrows():
        label = "cancer" if int(row["Diagnosis"]) == 1 else "healthy"
        print(f"    {row['Patient']}  true={label}  pred_mean={row['Prob']:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    train_rf_and_evaluate()
