"""
Quick test — ResNet-50 + RF on 2000 cells (1000 cancer, 1000 healthy).

Samples equally from each patient so no single patient dominates,
then applies the standard patient-grouped train/val split.

Run:  python quick_test_resnet.py
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
from sklearn.linear_model import Ridge
import joblib
from tqdm import tqdm
from resnet_extractor import ResNetExtractor

# ── Config ────────────────────────────────────────────────────────────────────
SEED            = 42
N_PER_CLASS     = 1000                                  # 1000 cancer + 1000 healthy
VAL_PATIENTS    = ["pat_07", "pat_10", "pat_05", "pat_16"]

SUB_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(SUB_ROOT)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

_normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
)


# ── Dataset ───────────────────────────────────────────────────────────────────

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
        img   = torch.cat([TF.to_tensor(bf), TF.to_tensor(fl)], dim=0)
        img   = _normalize(img)
        label = torch.tensor(float(row["Diagnosis"]), dtype=torch.float32)
        return img, label


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_balanced(df, n_per_class, seed):
    """
    Sample n_per_class cells from each class, distributed equally
    across patients within that class.
    """
    rng    = np.random.default_rng(seed)
    groups = []

    for label, group_df in df.groupby("Diagnosis"):
        patients   = group_df["Patient"].unique()
        per_patient = n_per_class // len(patients)
        remainder   = n_per_class  % len(patients)

        sampled = []
        for i, pat in enumerate(patients):
            pat_df = group_df[group_df["Patient"] == pat]
            n      = per_patient + (1 if i < remainder else 0)
            n      = min(n, len(pat_df))          # can't sample more than available
            sampled.append(pat_df.sample(n=n, random_state=int(rng.integers(1e6))))

        groups.append(pd.concat(sampled))

    return pd.concat(groups).sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_features(model, device, loader, name=""):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f"  [{name}]", leave=False):
            feats.append(model(imgs.to(device)).cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def safe_auc(y_true, y_score, name=""):
    if len(np.unique(y_true)) < 2:
        print(f"  [skip] {name}: only one class — AUC undefined")
        return None
    return float(roc_auc_score(y_true, y_score))


def _unique_label(s):
    v = s.unique()
    if len(v) != 1:
        raise ValueError(f"Inconsistent labels: {v}")
    return float(v[0])


def residualize_patient_effect(X_train, train_patients, X_val, val_patients):
    """
    Remove the linear component of patient identity from features.

    How it works
    ------------
    We fit a Ridge regression:  features ~ one_hot(patient_id)
    The model learns the average feature offset for each training patient.
    Subtracting those offsets leaves residuals that are orthogonal to
    patient identity — the RF then trains on patterns that cut across patients.

    Val patients are unseen, so we subtract only the global mean
    (the regression intercept), which is the best correction possible
    without knowing their patient-specific offset.
    """
    unique_train = np.unique(train_patients)

    # Manual one-hot: rows = cells, cols = training patients
    P_train = np.zeros((len(train_patients), len(unique_train)))
    for i, p in enumerate(unique_train):
        P_train[train_patients == p, i] = 1.0

    # Val patients are unknown → all-zero rows → Ridge predicts the intercept only
    P_val = np.zeros((len(val_patients), len(unique_train)))

    reg = Ridge(alpha=1.0)
    reg.fit(P_train, X_train)

    X_train_res = X_train - reg.predict(P_train)
    X_val_res   = X_val   - reg.predict(P_val)   # subtracts global mean

    return X_train_res, X_val_res


def leakage_diagnostic(X_train, patient_ids):
    le     = LabelEncoder()
    y      = le.fit_transform(patient_ids)
    n      = len(le.classes_)
    chance = 1.0 / n
    clf    = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    acc    = cross_val_score(clf, X_train, y, cv=5, scoring="accuracy").mean()
    ratio  = acc / chance
    print("\n" + "=" * 55)
    print("[Leakage] Patient-ID from ResNet features")
    print(f"  Chance: {chance:.3f}  |  CV acc: {acc:.3f}  |  {ratio:.1f}x chance")
    if   ratio > 3.0: print("  *** CRITICAL: strong patient identity in features")
    elif ratio > 2.0: print("  !! WARNING: moderate patient identity in features")
    else:             print("  OK: no strong patient-identity signal")
    print("=" * 55 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else "cpu"
    )
    print(f"\n[quick_test] device = {device}")

    # ── Sample 2000 cells ─────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    df["Patient"] = df["Name"].apply(lambda x: "_".join(x.split("_")[:2]))

    sampled = sample_balanced(df, n_per_class=N_PER_CLASS, seed=SEED)

    train_df = sampled[~sampled["Patient"].isin(VAL_PATIENTS)].reset_index(drop=True)
    val_df   = sampled[ sampled["Patient"].isin(VAL_PATIENTS)].reset_index(drop=True)

    print(f"\n[quick_test] Sampled {len(sampled)} cells total")
    print(f"  Cancer  : {int((sampled.Diagnosis==1).sum())}")
    print(f"  Healthy : {int((sampled.Diagnosis==0).sum())}")
    print(f"\n  Train split ({len(train_df)} cells):")
    print(train_df.groupby(["Patient","Diagnosis"]).size().to_string())
    print(f"\n  Val split ({len(val_df)} cells):")
    print(val_df.groupby(["Patient","Diagnosis"]).size().to_string())

    # ── Load model ────────────────────────────────────────────────────────────
    model = ResNetExtractor().to(device)
    model.eval()
    print("\n[quick_test] Frozen ResNet-50 loaded (ImageNet weights)")

    train_loader = DataLoader(CellDataset(train_df), batch_size=64,
                              shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(CellDataset(val_df),   batch_size=64,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Extract features ──────────────────────────────────────────────────────
    print("[quick_test] Extracting features…")
    X_train, y_train = extract_features(model, device, train_loader, "train")
    X_val,   y_val   = extract_features(model, device, val_loader,   "val")
    print(f"  Feature dim : {X_train.shape[1]}")

    # ── Leakage diagnostic ───────────────────────────────────────────────────
    leakage_diagnostic(X_train, train_df["Patient"].values)

    # ── RF ────────────────────────────────────────────────────────────────────
    print("[quick_test] Training RF…")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=4,
        random_state=SEED, n_jobs=-1, oob_score=True,
    )
    rf.fit(X_train, y_train)
    print(f"  OOB accuracy: {rf.oob_score_:.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_probs    = rf.predict_proba(X_val)[:, 1]
    val_cell_auc = safe_auc(y_val, val_probs, "val cell")

    val_df = val_df.copy()
    val_df["Prob"] = val_probs
    val_pat = (
        val_df.groupby("Patient")
        .agg(Prob=("Prob","mean"), Diagnosis=("Diagnosis", _unique_label))
        .reset_index()
    )
    val_pat_auc = safe_auc(val_pat["Diagnosis"].values, val_pat["Prob"].values, "val patient")

    print("\n" + "-" * 55)
    n = len(val_pat)
    print(f"Cell AUC    : {val_cell_auc:.4f}  [descriptive — cells not independent]")
    print(f"Patient AUC : {val_pat_auc:.4f}  [only {n} patients — statistically weak]")
    print("\nPer-patient:")
    for _, row in val_pat.iterrows():
        tag = "cancer" if int(row["Diagnosis"]) == 1 else "healthy"
        print(f"  {row['Patient']}  true={tag}  pred={row['Prob']:.4f}")
    print("-" * 55)


if __name__ == "__main__":
    main()
