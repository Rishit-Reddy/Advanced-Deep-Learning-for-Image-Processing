"""
Inference — ResNet-50 + RF pipeline.
Run after helper_resnet.py has saved best_rf_resnet.joblib.
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
from tqdm import tqdm
import joblib
from resnet_extractor import ResNetExtractor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SUB_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(SUB_ROOT)

_normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
)


class CellDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]["Name"]
        bf = Image.open(os.path.join(ROOT, "BF", "test", filename)).convert("RGB")
        fl = Image.open(os.path.join(ROOT, "FL", "test", filename)).convert("RGB")
        img = torch.cat([TF.to_tensor(bf), TF.to_tensor(fl)], dim=0)
        return _normalize(img)


def main():
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else "cpu"
    )
    print(f"[inference] device = {device}")

    model = ResNetExtractor().to(device)
    model.eval()
    print("[inference] Loaded frozen ResNet-50")

    rf_path = os.path.join(SUB_ROOT, "best_rf_resnet.joblib")
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"{rf_path} not found. Run helper_resnet.py first.")
    rf = joblib.load(rf_path)
    print(f"[inference] Loaded RF from {rf_path}")

    template_path = os.path.join(ROOT, "sampleSubmission.csv")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"{template_path} not found.")
    template_df = pd.read_csv(template_path)
    test_ds     = CellDataset(template_df.drop(columns=["Diagnosis"], errors="ignore"))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    print("[inference] Extracting ResNet features from test set…")
    all_feats = []
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="  test features"):
            all_feats.append(model(imgs.to(device)).cpu().numpy())
    X_test = np.concatenate(all_feats)

    template_df["Diagnosis"] = rf.predict_proba(X_test)[:, 1]
    out_path = os.path.join(SUB_ROOT, "submission_resnet_rf.csv")
    template_df.to_csv(out_path, index=False)
    print(f"[inference] Saved to {out_path}")


if __name__ == "__main__":
    main()
