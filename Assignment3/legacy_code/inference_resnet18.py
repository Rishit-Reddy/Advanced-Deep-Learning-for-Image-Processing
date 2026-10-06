import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF

from models_resnet18 import get_model

ROOT = os.path.dirname(os.path.abspath(__file__))


class CellDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]["Name"]
        bf = Image.open(os.path.join(ROOT, "BF/test", filename)).convert("RGB")
        fl = Image.open(os.path.join(ROOT, "FL/test", filename)).convert("RGB")
        bf_t = TF.to_tensor(bf)
        fl_t = TF.to_tensor(fl)
        return torch.cat([bf_t, fl_t], dim=0)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = os.path.join(ROOT, "best_model_resnet18_standalone.pt")
    model = get_model(model_path if os.path.exists(model_path) else None).to(device)
    if os.path.exists(model_path):
        print(f"Loaded {model_path} successfully.")
    else:
        print(f"Warning: {model_path} not found, using ImageNet-initialized ResNet18.")

    model.eval()

    template_df = pd.read_csv(os.path.join(ROOT, "sampleSubmission.csv"))
    test_ds = CellDataset(template_df)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Inference"):
            imgs = imgs.to(device)
            out = model(imgs)
            all_preds.append(torch.sigmoid(out).cpu().numpy())

    template_df["Diagnosis"] = np.concatenate(all_preds)
    template_df.to_csv(os.path.join(ROOT, "submission_resnet18.csv"), index=False)
    print("Inference complete. Created submission_resnet18.csv.")
