import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import time
from models_legacy import get_model
from tqdm import tqdm

# Use current directory as ROOT for cloud compatibility
ROOT = os.path.dirname(os.path.abspath(__file__))
VAL_PATIENTS = ["pat_07", "pat_10", "pat_05", "pat_16"]

class CellDataset(Dataset):
    def __init__(self, df, mode="train", augment=False):
        self.df = df
        self.mode = mode
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["Name"]
        label = row.get("Diagnosis", 0)

        dir_name = "train" if self.mode == "train" else "test"
        bf = Image.open(os.path.join(ROOT, f"BF/{dir_name}", filename)).convert("RGB")
        fl = Image.open(os.path.join(ROOT, f"FL/{dir_name}", filename)).convert("RGB")

        if self.augment:
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

        bf_t = TF.to_tensor(bf)
        fl_t = TF.to_tensor(fl)

        return torch.cat([bf_t, fl_t], dim=0), torch.tensor(label, dtype=torch.float32)

def run_inference(model, device, out_name="submission.csv"):
    print(f"\nGenerating predictions for test set using template from sampleSubmission.csv...")
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

def train_model(lr=1e-4, weight_decay=0, batch_size=128, samples_per_epoch=20000, epochs=15, exp_name="default"):
    print(f"\n--- Starting Experiment: {exp_name} ---")
    print(f"Params: LR={lr}, WD={weight_decay}, Batch={batch_size}, Samples={samples_per_epoch}, Epochs={epochs}")
    
    df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    df["Patient"] = df["Name"].apply(lambda x: '_'.join(x.split('_')[:2]))
    val_df = df[df["Patient"].isin(VAL_PATIENTS)].reset_index(drop=True)
    train_df = df[~df["Patient"].isin(VAL_PATIENTS)].reset_index(drop=True)
    
    train_ds = CellDataset(train_df, mode="train", augment=True)
    val_ds = CellDataset(val_df, mode="train", augment=False)
    
    labels = train_df["Diagnosis"].values
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in labels])
    sampler = WeightedRandomSampler(samples_weight, num_samples=samples_per_epoch, replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(os.path.join(ROOT, "DenseNet121.pt")).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    best_auc = 0.0
    epochs_without_improvement = 0
    early_stopping_patience = 5
    model_save_path = f"best_model_{exp_name}.pt"
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                all_preds.append(torch.sigmoid(out).cpu().numpy())
                all_labels.append(lbls.numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        auc = roc_auc_score(all_labels, all_preds)
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch+1}: New Best AUC {auc:.4f}!")
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= early_stopping_patience:
            print(f"--- Early stopping triggered for {exp_name} at epoch {epoch+1} ---")
            break
    
    print(f"Experiment {exp_name} Complete. Best AUC: {best_auc:.4f}")
    return best_auc

if __name__ == "__main__":
    # Default behavior: run one standard training session
    train_model(lr=1e-4, weight_decay=1e-5, samples_per_epoch=25000, epochs=15, exp_name="v1.2_standalone")
