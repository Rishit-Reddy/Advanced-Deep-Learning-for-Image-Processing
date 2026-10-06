
# Oral Cancer Cell Classification Project

## Project Overview
This project classifies oral cavity cells imaged in Brightfield (BF) and Fluorescence (FL) modalities using a DenseNet-121 architecture. Each cell is a 128x128 RGB image pair, combined into a 6-channel input.

## Core Mandates & Standards
- **Hardware:** Optimized for Apple M2 using `mps` device.
- **Level:** Assignment-level code (minimal exception handling, no logging frameworks, no docstrings/type hints).
- **Architecture:** DenseNet-121 from `torchvision`.
- **Pretrained Weights:** RadImageNet weights (`DenseNet121.pt`) loaded into the stem (`conv0`, `norm0`) and the first 3 dense blocks.
- **6-Channel Logic:** `conv0` is modified to accept 6 channels. Pretrained 3-channel weights are replicated across both modalities (BF and FL).
- **Training:** Entire network is trainable (no frozen layers).
- **Split:** Patient-grouped train/val split.
- **Validation Patients:** `pat_07`, `pat_09` (Healthy/0) and `pat_05` (Cancer/1).

## Data Layout
- `ROOT`: `/Users/rishitreddy/Projects/Uppsala_University/ADIP/Assignment3`
- `BF/train/`, `FL/train/`: Cell images (.jpg)
- `train.csv`: Labels (Name, Diagnosis)
- Input: 6-channel tensor (Channel 0-2: BF RGB, Channel 3-5: FL RGB).

## Current Workflow
1. **Research/Execution:** Fast turnaround using `SAMPLES_PER_EPOCH = 8000` with `WeightedRandomSampler` for class balance.
2. **Evaluation:** Cell-level AUC and patient-level mean probability aggregation.
3. **Hardware:** Uses `mps` if available, falling back to `cuda` or `cpu`.

## Files
- `models.py`: Model definition and RadImageNet weight mapping.
- `helper.py`: Dataset, training loop, and evaluation.
