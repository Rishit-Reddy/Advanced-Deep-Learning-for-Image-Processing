"""Data layer: CSV parsing, patient maps, 6-channel paired Dataset, splits.

A "cell" is one filename (e.g. pat_03_image_1.jpg). The same filename must
exist in both BF/train and FL/train; we skip and log any orphan filenames.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

import config


PATIENT_RE = re.compile(r"^(pat_\d+)_image_\d+\.jpg$")


# ----------------------------------------------------------------------
# CSV parsing and patient maps
# ----------------------------------------------------------------------
def parse_patient_id(filename: str) -> str:
    m = PATIENT_RE.match(filename)
    if not m:
        raise ValueError(f"Unrecognized filename pattern: {filename}")
    return m.group(1)


def load_train_index(
    csv_path: Path = config.TRAIN_CSV,
    bf_dir: Path = config.BF_TRAIN_DIR,
    fl_dir: Path = config.FL_TRAIN_DIR,
    verbose: bool = True,
):
    """Return (cells_df, patient_label, patient_to_cells, n_skipped).

    cells_df has columns [filename, patient, label] for cells present in
    BOTH BF and FL. Cells missing one modality are dropped.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Name": "filename", "Diagnosis": "label"})

    bf_set = {p.name for p in bf_dir.iterdir() if p.suffix.lower() == ".jpg"}
    fl_set = {p.name for p in fl_dir.iterdir() if p.suffix.lower() == ".jpg"}
    paired = bf_set & fl_set

    before = len(df)
    df = df[df["filename"].isin(paired)].copy()
    skipped = before - len(df)

    df["patient"] = df["filename"].map(parse_patient_id)
    df["label"] = df["label"].astype(np.float32)

    # Sanity: each patient has a single weak label.
    patient_label = {}
    for pat, sub in df.groupby("patient"):
        labels = sub["label"].unique()
        if len(labels) != 1:
            raise ValueError(f"Patient {pat} has inconsistent labels: {labels}")
        patient_label[pat] = float(labels[0])

    patient_to_cells = {pat: sub["filename"].tolist() for pat, sub in df.groupby("patient")}

    if verbose:
        print(f"[data] loaded {len(df)} paired cells across {len(patient_label)} patients")
        print(f"[data] skipped {skipped} cells (missing BF or FL modality)")
        for pat in sorted(patient_label):
            print(f"       {pat}: label={int(patient_label[pat])}, cells={len(patient_to_cells[pat])}")

    return df.reset_index(drop=True), patient_label, patient_to_cells, skipped


# ----------------------------------------------------------------------
# Augmentation primitives — applied identically to BF and FL
# ----------------------------------------------------------------------
def _sample_geom(rng: random.Random):
    """Sample one set of geometric transform params for a paired cell."""
    return {
        "hflip": config.AUG_FLIP and rng.random() < 0.5,
        "vflip": config.AUG_FLIP and rng.random() < 0.5,
        "angle": rng.uniform(-config.AUG_ROTATE_DEGREES, config.AUG_ROTATE_DEGREES)
                 if config.AUG_ROTATE else 0.0,
    }


def _apply_geom(img: torch.Tensor, p: dict) -> torch.Tensor:
    """img: [C,H,W] float in [0,1]."""
    if p["hflip"]:
        img = torch.flip(img, dims=[-1])
    if p["vflip"]:
        img = torch.flip(img, dims=[-2])
    if p["angle"] != 0.0:
        # Use a 2D affine grid for arbitrary rotation; reflect padding to
        # avoid black corners that would shift channel stats.
        c, h, w = img.shape
        theta = torch.tensor(np.deg2rad(p["angle"]), dtype=torch.float32)
        cos, sin = torch.cos(theta), torch.sin(theta)
        mat = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=torch.float32)
        grid = F.affine_grid(mat.unsqueeze(0), [1, c, h, w], align_corners=False)
        img = F.grid_sample(
            img.unsqueeze(0), grid, mode="bilinear",
            padding_mode="reflection", align_corners=False,
        ).squeeze(0)
    return img


def _photometric(img: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Per-modality brightness/contrast/color jitter; img is [3,H,W]."""
    if config.AUG_BRIGHTNESS > 0:
        factor = 1.0 + rng.uniform(-config.AUG_BRIGHTNESS, config.AUG_BRIGHTNESS)
        img = img * factor
    if config.AUG_CONTRAST > 0:
        factor = 1.0 + rng.uniform(-config.AUG_CONTRAST, config.AUG_CONTRAST)
        mean = img.mean(dim=(-2, -1), keepdim=True)
        img = (img - mean) * factor + mean
    if config.AUG_COLOR > 0:
        # Cheap per-channel multiplicative jitter as a color shift.
        scale = torch.tensor(
            [1.0 + rng.uniform(-config.AUG_COLOR, config.AUG_COLOR) for _ in range(img.shape[0])],
            dtype=img.dtype,
        ).view(-1, 1, 1)
        img = img * scale
    return img.clamp(0.0, 1.0)


def _noise(img: torch.Tensor, rng: random.Random) -> torch.Tensor:
    if config.AUG_NOISE and rng.random() < config.AUG_NOISE_PROB:
        img = img + torch.randn_like(img) * config.AUG_NOISE_STD
    return img.clamp(0.0, 1.0)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0  # [H,W,3]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]


class DualModalDataset(Dataset):
    """6-channel BF+FL dataset for one set of cells."""

    def __init__(
        self,
        filenames: Sequence[str],
        labels: Sequence[float] | None,
        bf_dir: Path,
        fl_dir: Path,
        train_aug: bool,
        normalize: bool = True,
    ):
        self.filenames = list(filenames)
        self.labels = list(labels) if labels is not None else None
        self.bf_dir = Path(bf_dir)
        self.fl_dir = Path(fl_dir)
        self.train_aug = train_aug
        self.normalize = normalize
        self._mean = torch.tensor(config.CHANNEL_MEAN, dtype=torch.float32).view(-1, 1, 1)
        self._std = torch.tensor(config.CHANNEL_STD, dtype=torch.float32).view(-1, 1, 1)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        bf = _load_rgb(self.bf_dir / name)
        fl = _load_rgb(self.fl_dir / name)

        if self.train_aug:
            # Per-sample RNG so DataLoader workers stay deterministic-ish per epoch
            # while still varying across samples.
            rng = random.Random()
            p = _sample_geom(rng)
            bf = _apply_geom(bf, p)
            fl = _apply_geom(fl, p)
            bf = _photometric(bf, rng)
            fl = _photometric(fl, rng)
            bf = _noise(bf, rng)
            fl = _noise(fl, rng)

        x = torch.cat([bf, fl], dim=0)  # [6,H,W]
        if self.normalize:
            x = (x - self._mean) / self._std

        if self.labels is None:
            return x, name
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ----------------------------------------------------------------------
# Patient-disjoint splits
# ----------------------------------------------------------------------
def build_split(
    val_patients: Iterable[str],
    cells_df: pd.DataFrame,
    train_aug: bool = True,
) -> tuple[DualModalDataset, DualModalDataset]:
    """Return (train_ds, val_ds) such that no patient appears in both."""
    val_set = set(val_patients)
    val_mask = cells_df["patient"].isin(val_set)
    train_df = cells_df[~val_mask]
    val_df = cells_df[val_mask]

    train_pats = set(train_df["patient"].unique())
    val_pats = set(val_df["patient"].unique())
    if train_pats & val_pats:
        raise AssertionError(f"Patient overlap in split: {train_pats & val_pats}")

    train_ds = DualModalDataset(
        train_df["filename"].tolist(), train_df["label"].tolist(),
        config.BF_TRAIN_DIR, config.FL_TRAIN_DIR, train_aug=train_aug,
    )
    val_ds = DualModalDataset(
        val_df["filename"].tolist(), val_df["label"].tolist(),
        config.BF_TRAIN_DIR, config.FL_TRAIN_DIR, train_aug=False,
    )
    return train_ds, val_ds


def build_full(cells_df: pd.DataFrame) -> DualModalDataset:
    """All 12 patients, train augs, used for final-mode submission training."""
    return DualModalDataset(
        cells_df["filename"].tolist(), cells_df["label"].tolist(),
        config.BF_TRAIN_DIR, config.FL_TRAIN_DIR, train_aug=True,
    )


def build_test(bf_dir: Path = config.BF_TEST_DIR, fl_dir: Path = config.FL_TEST_DIR) -> DualModalDataset:
    bf_set = {p.name for p in Path(bf_dir).iterdir() if p.suffix.lower() == ".jpg"}
    fl_set = {p.name for p in Path(fl_dir).iterdir() if p.suffix.lower() == ".jpg"}
    paired = sorted(bf_set & fl_set)
    skipped = (len(bf_set) + len(fl_set)) - 2 * len(paired)
    print(f"[data] test set: {len(paired)} paired cells, ~{skipped} unpaired skipped")
    return DualModalDataset(
        paired, None, bf_dir, fl_dir, train_aug=False,
    )


# ----------------------------------------------------------------------
# Per-channel normalization stats
# ----------------------------------------------------------------------
def compute_channel_stats(
    cells_df: pd.DataFrame,
    sample_per_patient: int | None = 500,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Compute mean/std for each of the 6 channels over (a sample of) train.

    Subsamples per patient to keep the pass fast; pass None to use all cells.
    """
    rng = np.random.default_rng(seed)
    if sample_per_patient is not None:
        rows = []
        for _, sub in cells_df.groupby("patient"):
            k = min(len(sub), sample_per_patient)
            rows.append(sub.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1))))
        sample_df = pd.concat(rows, ignore_index=True)
    else:
        sample_df = cells_df

    n = 0
    s = np.zeros(6, dtype=np.float64)
    s2 = np.zeros(6, dtype=np.float64)
    for fname in sample_df["filename"]:
        bf = _load_rgb(config.BF_TRAIN_DIR / fname).numpy()
        fl = _load_rgb(config.FL_TRAIN_DIR / fname).numpy()
        stacked = np.concatenate([bf, fl], axis=0)  # [6,H,W]
        flat = stacked.reshape(6, -1)
        s += flat.sum(axis=1)
        s2 += (flat ** 2).sum(axis=1)
        n += flat.shape[1]

    mean = s / n
    var = s2 / n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.tolist(), std.tolist()


# ----------------------------------------------------------------------
# Verification helpers (used as __main__)
# ----------------------------------------------------------------------
def verify_folds(folds=config.FOLDS, expected_patients=None, patient_label=None):
    """Print fold composition and assert partition / class-balance invariants."""
    flat = [p for fold in folds for p in fold]
    if len(flat) != len(set(flat)):
        dupes = [p for p in set(flat) if flat.count(p) > 1]
        raise AssertionError(f"Patient appears in multiple folds: {dupes}")

    if expected_patients is not None:
        miss = set(expected_patients) - set(flat)
        extra = set(flat) - set(expected_patients)
        if miss or extra:
            raise AssertionError(f"Fold partition mismatch: missing={miss}, extra={extra}")

    print(f"[verify] {len(folds)} folds, {len(flat)} patients, partition OK")
    for i, fold in enumerate(folds):
        if patient_label is not None:
            cancer = [p for p in fold if patient_label[p] == 1.0]
            healthy = [p for p in fold if patient_label[p] == 0.0]
            if not cancer or not healthy:
                raise AssertionError(f"Fold {i} lacks class balance: {fold}")
            print(f"  fold {i}: {fold}  cancer={cancer} healthy={healthy}")
        else:
            print(f"  fold {i}: {fold}")


def verify_split_disjoint(cells_df, fold_idx):
    train_ds, val_ds = build_split(config.FOLDS[fold_idx], cells_df)
    tp = {parse_patient_id(n) for n in train_ds.filenames}
    vp = {parse_patient_id(n) for n in val_ds.filenames}
    assert not (tp & vp), f"Fold {fold_idx}: patient overlap {tp & vp}"
    print(f"  fold {fold_idx}: train={len(train_ds)} cells / {len(tp)} pat, "
          f"val={len(val_ds)} cells / {len(vp)} pat  disjoint OK")


if __name__ == "__main__":
    cells, plabel, _, _ = load_train_index()
    print()
    verify_folds(expected_patients=sorted(plabel.keys()), patient_label=plabel)
    print()
    for i in range(len(config.FOLDS)):
        verify_split_disjoint(cells, i)
