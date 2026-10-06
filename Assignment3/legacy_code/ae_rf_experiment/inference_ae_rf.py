"""
Stage 3 — Generate test-set predictions using the trained AE + RF.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import joblib
from ae_models import ResidualAE

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
SUB_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(SUB_ROOT)


class CellDataset(Dataset):
    """Test-only dataset — loads images, no labels."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]["Name"]
        bf = Image.open(os.path.join(ROOT, "BF", "test", filename)).convert("RGB")
        fl = Image.open(os.path.join(ROOT, "FL", "test", filename)).convert("RGB")
        img = torch.cat([TF.to_tensor(bf), TF.to_tensor(fl)], dim=0)
        imin, imax = img.min(), img.max()
        img = (img - imin) / (imax - imin + 1e-6)
        return img


def main():
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else "cpu"
    )
    print(f"[inference] device = {device}")

    # Load AE
    ae_path = os.path.join(SUB_ROOT, "best_residual_ae.pt")
    if not os.path.exists(ae_path):
        raise FileNotFoundError(f"{ae_path} not found. Run train.py first.")
    ae = ResidualAE().to(device)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()
    print(f"[inference] Loaded AE from {ae_path}")

    # Load RF
    rf_path = os.path.join(SUB_ROOT, "best_rf_model.joblib")
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"{rf_path} not found. Run helper.py first.")
    rf = joblib.load(rf_path)
    print(f"[inference] Loaded RF from {rf_path}")

    # Load test set (drop Diagnosis placeholder column if present)
    template_path = os.path.join(ROOT, "sampleSubmission.csv")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"{template_path} not found.")
    template_df = pd.read_csv(template_path)
    test_ds     = CellDataset(template_df.drop(columns=["Diagnosis"], errors="ignore"))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # Extract features
    print("[inference] Extracting AE features from test set…")
    all_feats = []
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="  test features"):
            _, feat = ae(imgs.to(device))
            all_feats.append(feat.cpu().numpy())
    X_test = np.concatenate(all_feats)

    # Predict
    print("[inference] Generating RF predictions…")
    template_df["Diagnosis"] = rf.predict_proba(X_test)[:, 1]

    out_path = os.path.join(SUB_ROOT, "submission_ae_rf.csv")
    template_df.to_csv(out_path, index=False)
    print(f"[inference] Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
