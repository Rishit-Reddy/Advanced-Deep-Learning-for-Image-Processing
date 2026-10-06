import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import torchvision.transforms.functional as TF

from models_resnet18 import get_model

ROOT = os.path.dirname(os.path.abspath(__file__))
VAL_PATIENTS = ["pat_07", "pat_10", "pat_05", "pat_16"]


class CellDataset(Dataset):
    def __init__(self, df, mode="train", augment=False):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _apply_heavy_augmentation(self, bf, fl):
        if random.random() > 0.5:
            bf = TF.hflip(bf)
            fl = TF.hflip(fl)
        if random.random() > 0.5:
            bf = TF.vflip(bf)
            fl = TF.vflip(fl)

        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            bf = TF.rotate(bf, angle)
            fl = TF.rotate(fl, angle)

        brightness_factor = random.uniform(0.65, 1.35)
        contrast_factor = random.uniform(0.65, 1.35)
        saturation_factor = random.uniform(0.7, 1.3)
        hue_factor = random.uniform(-0.05, 0.05)
        bf = TF.adjust_brightness(bf, brightness_factor)
        fl = TF.adjust_brightness(fl, brightness_factor)
        bf = TF.adjust_contrast(bf, contrast_factor)
        fl = TF.adjust_contrast(fl, contrast_factor)
        bf = TF.adjust_saturation(bf, saturation_factor)
        fl = TF.adjust_saturation(fl, saturation_factor)
        bf = TF.adjust_hue(bf, hue_factor)
        fl = TF.adjust_hue(fl, hue_factor)

        return bf, fl

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["Name"]
        label = row.get("Diagnosis", 0)

        dir_name = "train" if self.mode == "train" else "test"
        bf = Image.open(os.path.join(ROOT, f"BF/{dir_name}", filename)).convert("RGB")
        fl = Image.open(os.path.join(ROOT, f"FL/{dir_name}", filename)).convert("RGB")

        if self.augment:
            bf, fl = self._apply_heavy_augmentation(bf, fl)

        bf_t = TF.to_tensor(bf)
        fl_t = TF.to_tensor(fl)

        if self.augment:
            # ~10% Gaussian noise as regularization, sampled independently per modality.
            bf_t = (bf_t + torch.randn_like(bf_t) * 0.1).clamp_(0.0, 1.0)
            fl_t = (fl_t + torch.randn_like(fl_t) * 0.1).clamp_(0.0, 1.0)

        return torch.cat([bf_t, fl_t], dim=0), torch.tensor(label, dtype=torch.float32)


def run_inference(model, device, out_name="submission_resnet18.csv"):
    print("\nGenerating predictions for the test set using sampleSubmission.csv...")
    template_path = os.path.join(ROOT, "sampleSubmission.csv")
    template_df = pd.read_csv(template_path)

    test_df = template_df[["Name"]].copy()
    test_ds = CellDataset(test_df, mode="test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Test Inference"):
            imgs = imgs.to(device)
            out = model(imgs)
            all_preds.append(torch.sigmoid(out).cpu().numpy())

    test_df["Diagnosis"] = np.concatenate(all_preds).flatten()
    test_df.to_csv(out_name, index=False)
    print(f"Saved {out_name}")


def _compute_metrics_from_logits(logits, labels, criterion):
    probabilities = torch.sigmoid(logits)
    loss = criterion(logits, labels.view(-1, 1)).item()
    predictions = (probabilities >= 0.5).float()
    accuracy = (predictions.view(-1) == labels.view(-1)).float().mean().item()
    return loss, accuracy, probabilities.cpu().numpy()


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(lr=1e-4, weight_decay=0, batch_size=32, samples_per_epoch=20000, epochs=25, exp_name="resnet18", seed=42):
    print(f"\n--- Starting Experiment: {exp_name} ---")
    print(f"Params: LR={lr}, WD={weight_decay}, Batch={batch_size}, Samples={samples_per_epoch}, Epochs={epochs}, Seed={seed}")

    _set_seed(seed)

    df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    df["Patient"] = df["Name"].apply(lambda x: "_".join(x.split("_")[:2]))
    val_df = df[df["Patient"].isin(VAL_PATIENTS)].reset_index(drop=True)
    train_df = df[~df["Patient"].isin(VAL_PATIENTS)].reset_index(drop=True)

    train_ds = CellDataset(train_df, mode="train", augment=True)
    val_ds = CellDataset(val_df, mode="train", augment=False)

    labels = train_df["Diagnosis"].values
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in labels])
    sampler = WeightedRandomSampler(samples_weight, num_samples=samples_per_epoch, replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    best_loss = float("inf")
    best_acc = 0.0
    best_auc = 0.0
    best_train_loss = 0.0
    best_train_acc = 0.0
    epochs_without_improvement = 0
    early_stopping_patience = 5
    model_save_path = f"best_model_{exp_name}.pt"

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0.0
        train_count = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls.view(-1, 1))
            loss.backward()
            optimizer.step()
            batch_size_actual = lbls.size(0)
            train_loss_sum += loss.item() * batch_size_actual
            train_probs = torch.sigmoid(out)
            train_pred = (train_probs >= 0.5).float().view(-1)
            train_correct += (train_pred == lbls.view(-1)).float().sum().item()
            train_count += batch_size_actual

        train_loss = train_loss_sum / max(train_count, 1.0)
        train_acc = train_correct / max(train_count, 1.0)

        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                val_logits.append(out.cpu())
                val_labels.append(lbls)

        val_logits = torch.cat(val_logits, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        val_loss, val_acc, val_probs = _compute_metrics_from_logits(val_logits, val_labels, criterion)
        auc = roc_auc_score(val_labels.numpy(), val_probs)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1} Summary: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, "
            f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Val AUC {auc:.4f}"
        )

        # Checkpoint and early-stopping both key off lowest val loss so every
        # reported metric describes the same saved model.
        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            best_auc = auc
            best_train_loss = train_loss
            best_train_acc = train_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch + 1}: New Best Val Loss {val_loss:.4f} (AUC {auc:.4f})!")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"--- Early stopping triggered for {exp_name} at epoch {epoch + 1} ---")
            break

    print(
        f"Experiment {exp_name} Complete. Best Val Loss: {best_loss:.4f}, "
        f"Best Val Acc: {best_acc:.4f}, Best Val AUC: {best_auc:.4f}"
    )
    return {
        "best_loss": best_loss,
        "best_acc": best_acc,
        "best_auc": best_auc,
        "train_loss": best_train_loss,
        "train_acc": best_train_acc,
    }


if __name__ == "__main__":
    train_model(lr=1e-4, weight_decay=1e-5, batch_size=32, samples_per_epoch=25000, epochs=25, exp_name="resnet18_standalone")
