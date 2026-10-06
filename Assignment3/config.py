"""Central configuration for the 6-channel dual-modality ResNet18 pipeline.

Everything that should be tweakable for an experiment lives here so that
data.py, model.py, experiment_runner.py and predict.py stay parameter-free.
"""

from pathlib import Path


# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

BF_TRAIN_DIR = ROOT / "BF" / "train"
FL_TRAIN_DIR = ROOT / "FL" / "train"
BF_TEST_DIR = ROOT / "BF" / "test"
FL_TEST_DIR = ROOT / "FL" / "test"

TRAIN_CSV = ROOT / "train.csv"
SAMPLE_SUBMISSION_CSV = ROOT / "sampleSubmission.csv"

CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
SUBMISSION_OUT = ROOT / "submission.csv"


# --- Cross-validation folds ---------------------------------------------
# 12 labeled patients: 5 cancer (pat_03/05/16/17/18) + 7 healthy
# (pat_07/09/10/11/13/14/15). k=4, each fold has 3 patients with at least
# one cancer and one healthy. The four val sets together partition all 12.
FOLDS = [
    ["pat_03", "pat_05", "pat_07"],   # 2 cancer, 1 healthy
    ["pat_16", "pat_09", "pat_10"],   # 1 cancer, 2 healthy
    ["pat_17", "pat_11", "pat_13"],   # 1 cancer, 2 healthy
    ["pat_18", "pat_14", "pat_15"],   # 1 cancer, 2 healthy
]

ALL_PATIENTS = sorted({p for fold in FOLDS for p in fold})


# --- Data / normalization ------------------------------------------------
IMG_SIZE = 128
NUM_CHANNELS = 6  # 3 BF + 3 FL stacked

# Per-channel mean/std across all 6 channels, computed once over the train
# set via data.compute_channel_stats. Placeholder values until that runs;
# experiment_runner refuses to start training while these are zero/one.
# Computed from ~500 cells/patient on the train set (data.compute_channel_stats).
CHANNEL_MEAN = [0.440022, 0.519699, 0.584218, 0.171709, 0.085172, 0.105391]
CHANNEL_STD = [0.258836, 0.203327, 0.170072, 0.142633, 0.202472, 0.166434]


# --- Augmentation toggles (train only) ----------------------------------
AUG_FLIP = True
AUG_ROTATE = True
AUG_ROTATE_DEGREES = 180.0          # uniform in [-deg, +deg]
AUG_JITTER = False
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
AUG_COLOR = 0.1                      # hue-ish jitter on RGB
AUG_NOISE = True
AUG_NOISE_PROB = 0.10
AUG_NOISE_STD = 0.02


# --- Training hyperparameters -------------------------------------------
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 8

LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_P = 0.5
FREEZE_EARLY_BLOCKS = True

EPOCH_CAP = 30
EARLY_STOP_PATIENCE = 6              # epochs without val-loss improvement
USE_POS_WEIGHT = True                # weight the cancer class by inverse freq

# Final-mode all-12 retrain budget. There is no val set in final mode, so
# this is a fixed number derived from where the CV folds usually checkpoint.
# Re-tune after looking at the per-fold "best epoch" prints.
FINAL_EPOCHS = 12


# --- Device --------------------------------------------------------------
# "cuda" if available, else "mps" on Apple Silicon, else "cpu".
def resolve_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
