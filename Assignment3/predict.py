"""Inference on the test set.

Loads the final all-12 checkpoint, predicts per-cell probabilities, writes
a submission CSV with the same header/order as sampleSubmission.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
import data
import model as model_mod


@torch.no_grad()
def predict(ckpt_path: Path, out_csv: Path, batch_size: int = 128):
    device = config.resolve_device()
    print(f"[predict] device={device}")
    print(f"[predict] loading {ckpt_path}")

    net = model_mod.build_model().to(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)
    net.eval()

    test_ds = data.build_test()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)

    names, probs = [], []
    for x, batch_names in loader:
        x = x.to(device, non_blocking=True)
        logits = net(x).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        names.extend(list(batch_names))
        probs.extend(p.tolist())

    pred_df = pd.DataFrame({"Name": names, "Diagnosis": probs})

    # Match the header/order of sampleSubmission.csv exactly.
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_CSV)
    merged = sample[["Name"]].merge(pred_df, on="Name", how="left")
    missing = int(merged["Diagnosis"].isna().sum())
    if missing:
        print(f"[predict] WARN: {missing} sample rows had no prediction; filling 0.5")
        merged["Diagnosis"] = merged["Diagnosis"].fillna(0.5)

    merged.to_csv(out_csv, index=False)
    print(f"[predict] wrote {len(merged)} rows -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=config.CHECKPOINT_DIR / "final.pt")
    ap.add_argument("--out", type=Path, default=config.SUBMISSION_OUT)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    predict(args.ckpt, args.out, args.batch_size)


if __name__ == "__main__":
    main()
