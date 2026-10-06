"""
Stage 1 — Train the denoising autoencoder (ResidualAE).

Split design
------------
AE trains on  : labeled train patients (excluding VAL_PATIENTS and AE_STOP_PATIENT)
                + unlabeled test images if sampleSubmission.csv is present (self-supervised,
                  so including them is legitimate and improves generalisation).
AE early-stops: reconstruction loss on AE_STOP_PATIENT cells only. No label is used here,
                so this patient can still be used for RF training in helper.py without
                introducing any label-level leakage.
RF trains on  : all non-val patients (helper.py), which includes AE_STOP_PATIENT.
"""

import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, RandomSampler
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from ae_models import ResidualAE

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
SUB_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(SUB_ROOT)

# ── Patient splits ───────────────────────────────────────────────────────────
VAL_PATIENTS = ["pat_07", "pat_10", "pat_05", "pat_16"]

# One training patient withheld from AE training; used ONLY for early-stopping
# via reconstruction loss (no label information used). Still included in RF
# training in helper.py because no label-driven decision touches it here.
AE_STOP_PATIENT = "pat_03"

assert AE_STOP_PATIENT not in VAL_PATIENTS, \
    f"AE_STOP_PATIENT '{AE_STOP_PATIENT}' must not be in VAL_PATIENTS"


class CellDataset(Dataset):
    """Loads paired BF+FL images. Returns label=-1 for unlabeled rows."""

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
        img = torch.cat([TF.to_tensor(bf), TF.to_tensor(fl)], dim=0)  # 6×128×128

        # Per-image min-max normalization: maps each image to [0,1] based on its
        # own range, removing absolute brightness/staining differences between
        # patients without breaking the decoder's Sigmoid output.
        imin, imax = img.min(), img.max()
        img = (img - imin) / (imax - imin + 1e-6)

        # AE ignores labels; -1 is a safe sentinel for unlabeled rows.
        if "Diagnosis" in row.index and not pd.isna(row["Diagnosis"]):
            label = float(row["Diagnosis"])
        else:
            label = -1.0

        return img, torch.tensor(label, dtype=torch.float32)


def train_ae(samples_per_epoch=25_000, max_epochs=50, patience=5, batch_size=128):
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else "cpu"
    )
    print(f"[train] device = {device}")

    # ── Build splits ─────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    df["Patient"] = df["Name"].apply(lambda x: "_".join(x.split("_")[:2]))

    if AE_STOP_PATIENT not in df["Patient"].unique():
        raise ValueError(
            f"AE_STOP_PATIENT='{AE_STOP_PATIENT}' not found in train.csv. "
            "Update it to a patient that exists in the labeled training data."
        )

    excluded    = set(VAL_PATIENTS) | {AE_STOP_PATIENT}
    ae_train_df = df[~df["Patient"].isin(excluded)].reset_index(drop=True)
    ae_stop_df  = df[df["Patient"] == AE_STOP_PATIENT].reset_index(drop=True)

    print(f"[train] AE train patients : {sorted(ae_train_df['Patient'].unique())}")
    print(f"[train] AE stop patient   : {AE_STOP_PATIENT}  ({len(ae_stop_df)} cells)")

    # ── Build datasets ────────────────────────────────────────────────────────
    ae_train_ds = CellDataset(ae_train_df, split="train")

    # Include unlabeled test images in AE training (self-supervised: no label needed).
    test_csv = os.path.join(ROOT, "sampleSubmission.csv")
    if os.path.exists(test_csv):
        test_df_unlabeled = pd.read_csv(test_csv).drop(columns=["Diagnosis"], errors="ignore")
        ae_train_ds = ConcatDataset([ae_train_ds, CellDataset(test_df_unlabeled, split="test")])
        print(f"[train] Added {len(test_df_unlabeled)} unlabeled test images to AE training.")
    else:
        print("[train] sampleSubmission.csv not found — AE trains on labeled data only.")

    ae_stop_ds = CellDataset(ae_stop_df, split="train")

    # ── Samplers & loaders ────────────────────────────────────────────────────
    # AE is fully self-supervised — class balance is irrelevant.
    # Use a seeded RandomSampler to control epoch length and stay reproducible.
    g       = torch.Generator().manual_seed(SEED)
    sampler = RandomSampler(ae_train_ds, num_samples=samples_per_epoch,
                            replacement=True, generator=g)

    train_loader = DataLoader(ae_train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    stop_loader  = DataLoader(ae_stop_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = ResidualAE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_stop_loss   = float("inf")
    epochs_no_improv = 0
    model_path       = os.path.join(SUB_ROOT, "best_residual_ae.pt")

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [train]", leave=False):
            imgs = imgs.to(device)

            # Random spatial augmentation — breaks patient-specific slide orientation
            # and spatial patterns. Applied before noise so the clean target is also augmented.
            imgs = torch.rot90(imgs, random.randint(0, 3), dims=[2, 3])
            if random.random() > 0.5:
                imgs = torch.flip(imgs, dims=[3])  # horizontal flip
            if random.random() > 0.5:
                imgs = torch.flip(imgs, dims=[2])  # vertical flip

            # Gaussian denoising noise (σ=0.3); stronger corruption forces more robust features.
            noisy = torch.clamp(imgs + torch.randn_like(imgs) * 0.3, 0.0, 1.0)
            optimizer.zero_grad()
            recon, _ = model(noisy)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # Early-stopping check on AE_STOP_PATIENT clean reconstruction (no labels used).
        model.eval()
        stop_loss = 0.0
        with torch.no_grad():
            for imgs, _ in stop_loader:
                imgs = imgs.to(device)
                recon, _ = model(imgs)
                stop_loss += criterion(recon, imgs).item()
        avg_stop = stop_loss / len(stop_loader)

        print(f"Epoch {epoch+1:3d}: train_recon={avg_train:.6f}  stop_recon={avg_stop:.6f}")

        if avg_stop < best_stop_loss:
            best_stop_loss   = avg_stop
            epochs_no_improv = 0
            torch.save(model.state_dict(), model_path)
            print(f"           >>> best model saved (stop_loss={best_stop_loss:.6f})")
        else:
            epochs_no_improv += 1

        if epochs_no_improv >= patience:
            print(f"[train] Early stopping at epoch {epoch+1}.")
            break

    print(f"[train] AE saved to {model_path}")
    return model_path


if __name__ == "__main__":
    train_ae(samples_per_epoch=25_000, max_epochs=50, patience=5, batch_size=128)
