# Multimodal Cancer Classification

Advanced Deep Learning for Image Processing, Uppsala University, MSc. A Kaggle-hosted
course challenge, done as a team of five (Group 3). We placed ninth.

## The problem

Predict whether a single cell image comes from a cancer patient or a healthy one. The
cells come from oral brush scrapes, stained and imaged two ways, brightfield (BF) and
fluorescence (FL), as 128 by 128 crops. There are only 19 patients, and every cell
carries its patient's diagnosis as its label, so plenty of cells labelled as cancer
look completely normal. The real question was never how well a model fits the data, it
was how well it works on a patient it has never seen.

## What's in here

The code that runs end to end is the 6-channel ResNet-18 pipeline: BF and FL are
stacked into one 6-channel image and fed to a single ImageNet-pretrained ResNet-18
whose first conv is widened from 3 to 6 channels. Everything is split by patient, never
by cell.

| File | What it does |
|---|---|
| `config.py` | Every knob in one place: paths, the four patient folds, channel mean/std, augmentation toggles, learning rate, early stopping, device pick. The other files read from here and take no parameters of their own. |
| `data.py` | Reads `train.csv`, maps each filename to its patient, pairs the BF and FL crop into one 6-channel tensor, applies the same geometric augmentation to both, and builds the train/val/test splits. Also computes the channel statistics in `config.py`. |
| `model.py` | Builds the model. Duplicates the pretrained 3-channel conv1 weights and halves them so activations start at roughly the right scale, swaps in a dropout + single-logit head, and freezes conv1/bn1/layer1 (keeping their BatchNorm pinned to eval, which is the part that is easy to get wrong). |
| `experiment_runner.py` | Training. `--mode fold --fold N` trains one fold, `--mode all` trains all four from scratch and aggregates, `--mode final` retrains on all 12 patients for a fixed number of epochs for the submission. Class weighting, early stopping, per-epoch logs. |
| `predict.py` | Loads a checkpoint, predicts per-cell probabilities, writes a submission CSV in the same row order as `sampleSubmission.csv`. |
| `models_resnet18.py`, `helper_legacy.py` | The earlier standalone version of the same model and its training helpers, before the config/data/model split. |
| `Data_Analysis.ipynb` | Looking at the data: class balance, cells per patient, what BF and FL actually look like. |
| `legacy_code/` | Things we tried and moved on from. DenseNet-121 (`helper.py`, `inference.py`, `results.md`), the ResNet-18 runs (`helper_resnet18.py`), a hyperparameter sweep (`experiments_summary.md`), and `ae_rf_experiment/`, an autoencoder plus random forest that scored 0.5630 and told us the features weren't there. |
| `submission*.csv`, `submit.csv` | Submissions, including the best one. |

The two-stream cross-attention model that gave our best score isn't in this repo; it was
run on a teammate's side and only its submission file landed here.

Not committed: `BF/` and `FL/` (the images, about 1.3 GB), checkpoints, logs.

## Running it

```bash
python experiment_runner.py --mode all     # four patient folds
python experiment_runner.py --mode final   # retrain on all 12 patients
python predict.py --ckpt checkpoints/final.pt --out submission.csv
```

Images go in `BF/train`, `FL/train`, `BF/test`, `FL/test`, with matching filenames
across the two.

## The split, and why it is the whole assignment

If you split cells at random, the same patient ends up in both training and validation
and the AUC looks great for the wrong reason. So we split by patient, which left a tiny
validation set that swung depending on who was held out. The four folds in `config.py`
each hold out three patients and together cover all twelve, with at least one cancer and
one healthy patient per fold.

I also trained a CNN to predict patient ID among the 12. It did far better than chance,
which told me identity is strongly encoded in these images. Heavier augmentation only
reduced it partly.

## Results (AUC)

| Model | Train | Validation | Test (public) |
|---|---|---|---|
| CNN auto-encoder + random forest | – | – | 0.5630 |
| DenseNet-121 | 0.9976 | 0.9230 | 0.6923 |
| 4-fold ensemble, ResNet-18 | 0.9843 | 0.9023 | 0.7319 |
| Two-stream ResNet + FCNN, transfer learning (ResNet-50) | 0.9878 | 0.7893 | 0.6594 |
| Two-stream ResNet + FCNN, fine-tuning | 0.9994 | 0.8235 | 0.7436 |
| Two-stream ResNet + cross-attention, transfer learning (ResNet-50) | 0.9963 | 0.7592 | 0.7027 |
| Two-stream ResNet + cross-attention, fine-tuning | 0.9999 | 0.9135 | 0.7788 |
| **Two-stream ResNet + multiscale cross-attention, fine-tuning** | **0.9999** | **0.9135** | **0.7818** |
| Two-stream DenseNet + multiscale cross-attention, transfer learning | 0.9992 | 0.9048 | 0.7414 |

Ninth place. The top score was 0.8651, and the challenge aimed for 0.85. Fine-tuning beat
a frozen backbone and test-time augmentation helped. Ensembling folds and training longer
didn't.

## What went wrong

The gap between 0.91 and 0.78 says it all: with so few patients, my validation score
wasn't something I could trust. I also spent too much time swapping backbones when I
should have been thinking about where to fuse the two images. The group that topped the
challenge used early fusion, combining the two images before the network, while our
fusion happened after the two backbones. And I didn't write down what I tried, so I
couldn't even learn from it properly. That's why I document failures now.
