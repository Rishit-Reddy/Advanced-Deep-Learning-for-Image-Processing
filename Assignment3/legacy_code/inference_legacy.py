import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from models_legacy import get_model
from tqdm import tqdm

ROOT = "/Users/rishitreddy/Projects/Uppsala_University/ADIP/Assignment3"

class CellDataset(Dataset):
    def __init__(self, df):
        self.df = df
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

    # Load model and weights
    model = get_model(os.path.join(ROOT, "DenseNet121.pt"))
    if os.path.exists("best_model_optuna_trial_1_bs32.pt"):
        # Load to CPU first to avoid storage mapping issues with MPS
        state_dict = torch.load("best_model_optuna_trial_1_bs32.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(device)
        print("Loaded best_model_optuna_trial_1_bs32.pt successfully.")
    else:
        print("Error: best_model_optuna_trial_1_bs32.pt not found.")
        exit(1)
        
    model.eval()

    # Prepare template-based dataframe
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
    template_df.to_csv("submission.csv", index=False)
    print("Inference complete. Created submission.csv with correct ordering.")
