"""Entry point for CV-fold and final-mode training.

Modes
-----
  python experiment_runner.py --mode fold --fold 0
  python experiment_runner.py --mode all
  python experiment_runner.py --mode final

In --mode all, each fold trains a freshly initialized model; metrics are
aggregated across folds. Weights are NEVER carried between folds.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import config
import data
import model as model_mod


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )


def compute_pos_weight(labels) -> torch.Tensor:
    arr = np.asarray(labels, dtype=np.float32)
    pos = float(arr.sum())
    neg = float(len(arr) - pos)
    if pos <= 0 or neg <= 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)


# ----------------------------------------------------------------------
# Train / eval epoch
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, verbose=False, batch_log_interval=10):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    n_batches = len(loader)
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).sum().item()

        # Always print batch progress (default every 10 batches)
        if (batch_idx + 1) % batch_log_interval == 0 or (batch_idx + 1) == n_batches:
            avg_loss = total_loss / total
            avg_acc = total_correct / total
            pct = 100.0 * (batch_idx + 1) / n_batches
            print(f"           [{batch_idx + 1:>4d}/{n_batches:<4d} {pct:>5.1f}%] loss {avg_loss:.4f} acc {avg_acc:.3f}")

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, filenames=None, batch_log_interval=10):
    """Return (loss, acc, auc, per_patient_dict, all_preds, all_targets)."""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    preds_all, y_all = [], []
    n_batches = len(loader)
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs
        total_correct += ((probs > 0.5).float() == y).sum().item()
        preds_all.append(probs.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

        # Batch progress during validation
        if (batch_idx + 1) % batch_log_interval == 0 or (batch_idx + 1) == n_batches:
            avg_loss = total_loss / total
            avg_acc = total_correct / total
            pct = 100.0 * (batch_idx + 1) / n_batches
            print(f"           [{batch_idx + 1:>4d}/{n_batches:<4d} {pct:>5.1f}%] loss {avg_loss:.4f} acc {avg_acc:.3f}")

    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)
    try:
        auc = roc_auc_score(y_all, preds_all)
    except ValueError:
        auc = float("nan")

    per_patient = {}
    if filenames is not None:
        # Group by patient: mean prob and (if both classes present) AUC.
        groups = defaultdict(list)
        for f, p, t in zip(filenames, preds_all, y_all):
            groups[data.parse_patient_id(f)].append((p, t))
        for pat, items in sorted(groups.items()):
            p_arr = np.array([x[0] for x in items])
            t_arr = np.array([x[1] for x in items])
            mean_p = float(p_arr.mean())
            label = float(t_arr[0])
            per_patient[pat] = {"label": label, "mean_pred": mean_p, "n": len(items)}

    return total_loss / max(total, 1), total_correct / max(total, 1), float(auc), per_patient, preds_all, y_all


# ----------------------------------------------------------------------
# One-fold trainer
# ----------------------------------------------------------------------
def train_fold(fold_idx: int, cells_df, device, ckpt_dir: Path, log_dir: Path, verbose: bool = True):
    set_seed(config.SEED + fold_idx)
    val_pats = config.FOLDS[fold_idx]
    train_ds, val_ds = data.build_split(val_pats, cells_df, train_aug=True)

    print(f"\n{'='*80}")
    print(f"Fold {fold_idx}: val_patients={val_pats}")
    print(f"  train: {len(train_ds)} cells from {len(set(data.parse_patient_id(n) for n in train_ds.filenames))} patients")
    print(f"  val:   {len(val_ds)} cells from {len(set(data.parse_patient_id(n) for n in val_ds.filenames))} patients")
    print(f"{'='*80}")

    train_loader = make_loader(train_ds, config.BATCH_SIZE, shuffle=True,
                               num_workers=config.NUM_WORKERS)
    val_loader = make_loader(val_ds, config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS)

    print(f"  DataLoader: batch_size={config.BATCH_SIZE}, num_workers={config.NUM_WORKERS}")
    print(f"  Train batches per epoch: {len(train_loader)}")
    print(f"  Val batches per epoch: {len(val_loader)}")

    net = model_mod.build_model().to(device)
    n_trainable = sum(p.numel() for p in model_mod.trainable_parameters(net))
    n_total = sum(p.numel() for p in net.parameters())
    print(f"  Model: {n_trainable:,} trainable / {n_total:,} total parameters")
    print(f"  Freeze early blocks: {config.FREEZE_EARLY_BLOCKS}")

    pos_weight = compute_pos_weight(train_ds.labels) if config.USE_POS_WEIGHT else None
    if pos_weight is not None:
        print(f"  Using pos_weight={pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = torch.optim.AdamW(
        model_mod.trainable_parameters(net),
        lr=config.LR, weight_decay=config.WEIGHT_DECAY,
    )
    print(f"  Optimizer: AdamW lr={config.LR} weight_decay={config.WEIGHT_DECAY}")
    print(f"  Early stopping: patience={config.EARLY_STOP_PATIENCE} epochs, cap={config.EPOCH_CAP} epochs")

    best_val_loss = math.inf
    best_val_auc = float("nan")
    best_epoch = -1
    patience = 0
    history = []

    ckpt_path = ckpt_dir / f"fold{fold_idx}_best.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    early_stopped = False
    print(f"\n  Training... ({config.EPOCH_CAP} epochs)")
    for epoch in range(1, config.EPOCH_CAP + 1):
        t0 = time.time()
        print(f"    [{epoch:02d}/{config.EPOCH_CAP:02d}] train phase:")
        tr_loss, tr_acc = train_one_epoch(net, train_loader, criterion, optimizer, device, verbose=verbose)
        print(f"    [{epoch:02d}/{config.EPOCH_CAP:02d}] val phase:")
        val_loss, val_acc, val_auc, per_pat, _, _ = eval_epoch(
            net, val_loader, criterion, device, filenames=val_ds.filenames,
        )
        dt = time.time() - t0

        print(f"  [{epoch:02d}/{config.EPOCH_CAP:02d}] train: loss {tr_loss:.4f} acc {tr_acc:.3f} "
              f"| val: loss {val_loss:.4f} acc {val_acc:.3f} auc {val_auc:.4f} | {dt:.1f}s")
        for pat in sorted(per_pat):
            d = per_pat[pat]
            label = "cancer" if int(d['label']) == 1 else "healthy"
            print(f"         {pat:8s} ({label:7s}) pred={d['mean_pred']:.4f} n={d['n']:>5d}")

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_auc": val_auc,
            "per_patient": per_pat,
        })

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            patience = 0
            torch.save(net.state_dict(), ckpt_path)
            print(f"         ✓ CHECKPOINT saved (val_loss {val_loss:.4f}, auc {val_auc:.4f})")
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                early_stopped = True
                print(f"         ✗ EARLY STOP (no improvement for {patience} epochs)")
                break

    end_reason = "early-stopped" if early_stopped else "epoch-cap reached"
    print(f"\n  Summary: best epoch={best_epoch} ({end_reason})")
    print(f"    val_loss={best_val_loss:.4f}, val_auc={best_val_auc:.4f}")

    with (log_dir / f"fold{fold_idx}_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    return {
        "fold": fold_idx, "best_epoch": best_epoch,
        "best_val_loss": best_val_loss, "best_val_auc": best_val_auc,
        "early_stopped": early_stopped,
    }


# ----------------------------------------------------------------------
# Final-mode trainer (all 12 patients, no val)
# ----------------------------------------------------------------------
def train_final(cells_df, device, ckpt_dir: Path, verbose: bool = True):
    set_seed(config.SEED + 999)
    ds = data.build_full(cells_df)
    n_patients = len(set(data.parse_patient_id(n) for n in ds.filenames))

    print(f"\n{'='*80}")
    print(f"FINAL MODE: training on all {n_patients} patients")
    print(f"  cells: {len(ds)}")
    print(f"  epochs: {config.FINAL_EPOCHS} (fixed budget, no validation)")
    print(f"{'='*80}")

    loader = make_loader(ds, config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    print(f"  DataLoader: batch_size={config.BATCH_SIZE}, batches_per_epoch={len(loader)}")

    net = model_mod.build_model().to(device)
    n_trainable = sum(p.numel() for p in model_mod.trainable_parameters(net))
    n_total = sum(p.numel() for p in net.parameters())
    print(f"  Model: {n_trainable:,} trainable / {n_total:,} total parameters")

    pos_weight = compute_pos_weight(ds.labels) if config.USE_POS_WEIGHT else None
    if pos_weight is not None:
        print(f"  Using pos_weight={pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = torch.optim.AdamW(
        model_mod.trainable_parameters(net),
        lr=config.LR, weight_decay=config.WEIGHT_DECAY,
    )
    print(f"  Optimizer: AdamW lr={config.LR} weight_decay={config.WEIGHT_DECAY}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "final.pt"

    print(f"\n  Training... ({config.FINAL_EPOCHS} epochs)")
    for epoch in range(1, config.FINAL_EPOCHS + 1):
        t0 = time.time()
        print(f"    [{epoch:02d}/{config.FINAL_EPOCHS:02d}] train phase:")
        loss, acc = train_one_epoch(net, loader, criterion, optimizer, device, verbose=verbose)
        dt = time.time() - t0
        print(f"  [{epoch:02d}/{config.FINAL_EPOCHS:02d}] train: loss {loss:.4f} acc {acc:.3f} | {dt:.1f}s")

    torch.save(net.state_dict(), ckpt_path)
    print(f"\n  ✓ Checkpoint saved: {ckpt_path}")
    return ckpt_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fold", "all", "final"], required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="Print per-batch training progress")
    args = ap.parse_args()

    device = config.resolve_device()
    print(f"\n{'='*80}")
    print(f"EXPERIMENT RUNNER")
    print(f"{'='*80}")
    print(f"device:        {device}")
    print(f"mode:          {args.mode}")
    if args.mode == "fold":
        print(f"fold_index:    {args.fold}")
    print(f"verbose:       {args.verbose}")
    print(f"seed:          {config.SEED}")
    print(f"batch_size:    {config.BATCH_SIZE}")
    print(f"lr:            {config.LR}")
    print(f"weight_decay:  {config.WEIGHT_DECAY}")
    print(f"dropout_p:     {config.DROPOUT_P}")
    print(f"epoch_cap:     {config.EPOCH_CAP}")
    print(f"early_stop_patience: {config.EARLY_STOP_PATIENCE}")

    print()
    cells_df, patient_label, _, _ = data.load_train_index(verbose=False)
    data.verify_folds(expected_patients=sorted(patient_label.keys()),
                      patient_label=patient_label)

    ckpt_dir = config.CHECKPOINT_DIR
    log_dir = config.LOG_DIR

    if args.mode == "fold":
        train_fold(args.fold, cells_df, device, ckpt_dir, log_dir, verbose=args.verbose)

    elif args.mode == "all":
        results = []
        for k in range(len(config.FOLDS)):
            r = train_fold(k, cells_df, device, ckpt_dir, log_dir, verbose=args.verbose)
            results.append(r)

        print(f"\n{'='*80}")
        print("CV SUMMARY")
        print(f"{'='*80}")
        aucs = []
        for r in results:
            mode = "early-stop" if r['early_stopped'] else "epoch-cap"
            print(f"  fold {r['fold']}: best_epoch={r['best_epoch']:>2d}  "
                  f"val_loss={r['best_val_loss']:.4f}  val_auc={r['best_val_auc']:.4f}  ({mode})")
            if not math.isnan(r["best_val_auc"]):
                aucs.append(r["best_val_auc"])
        if aucs:
            mean = float(np.mean(aucs))
            std = float(np.std(aucs, ddof=0))
            print(f"\n  MEAN AUC: {mean:.4f} ± {std:.4f}")
            print(f"  (based on {len(aucs)} folds)")
        with (log_dir / "cv_summary.json").open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {log_dir / 'cv_summary.json'}")

    elif args.mode == "final":
        train_final(cells_df, device, ckpt_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
