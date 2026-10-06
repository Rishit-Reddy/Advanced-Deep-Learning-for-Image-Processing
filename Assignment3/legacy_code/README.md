PyTorch training script for DenseNet121 (uses model_versions.model_v0)

Quick start:

1. Install requirements (recommended in a venv):

```bash
pip install torch torchvision tqdm pillow
```

2. If you have a CSV `train.csv` with columns `image,label` (labels 0/1), run:

```bash
python train_pytorch.py --data-csv train.csv --data-dir . --checkpoint DenseNet121.pt --epochs 10 --batch-size 16
```

3. If you don't have a CSV, place images under class subfolders and run:

```bash
python train_pytorch.py --data-dir path/to/data --checkpoint DenseNet121.pt
```

This script prints per-epoch summaries and uses `tqdm` to show verbose batch-level progress.
