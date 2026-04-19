
import os
import random
import time
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader, Subset

def train(model, train_loader, val_loader, epochs=50, lr=1e-3, weight_decay=1e-3,
          use_scheduler=False, scheduler_factor=0.5, scheduler_patience=5):
    """Train the model and keep the weights from the best validation Dice."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    best_val_dice = float('-inf')
    best_train_dice = float('-inf')
    best_epoch = -1
    best_model_state = None

    starttime = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            epoch_dice += dice_score(labels.cpu(), (outputs.cpu() > 0).float()) * images.size(0)

        train_losses.append(epoch_loss / len(train_loader.dataset))
        train_dices.append(epoch_dice / len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_dice += dice_score(labels.cpu(), (outputs.cpu() > 0).float()) * images.size(0)

        val_losses.append(val_loss / len(val_loader.dataset))
        val_dices.append(val_dice / len(val_loader.dataset))

        if scheduler is not None:
            scheduler.step(val_losses[-1])

        if val_dices[-1] > best_val_dice:
            # save a cpu copy so we can reload later without touching the gpu state
            best_val_dice = val_dices[-1]
            best_train_dice = train_dices[-1]
            best_epoch = epoch + 1
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Train Dice: {train_dices[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Dice: {val_dices[-1]:.4f}, LR: {current_lr:.2e}')

    training_time = time.time() - starttime

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, train_dices, val_losses, val_dices, training_time, best_epoch, best_train_dice, best_val_dice

def run_experiment(model_cls, train_loader, val_loader, title, model_path,
                   epochs=200, lr=1e-3, batch_size=16, weight_decay=1e-3,
                   test_loader=None, curve_split_name="test", use_scheduler=False,
                   scheduler_factor=0.5, scheduler_patience=5):
    """Train (or reload) a model, then plot learning curves and best/worst predictions."""

    metrics_path = f"{model_path}.metrics.pt"
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Train set size: {len(train_loader.dataset)} samples')
    print(f'Validation set size: {len(val_loader.dataset)} samples')
    eval_loader = test_loader if test_loader is not None else val_loader
    eval_split_name = 'test' if test_loader is not None else 'validation'
    prediction_summary = None

    if os.path.exists(model_path):
        # skip training if we already have a saved model
        print(f'Loading existing model from {model_path}, skipping training.')
        model = model_cls().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        if os.path.exists(metrics_path):
            best_metrics = torch.load(metrics_path, map_location='cpu')
            best_val = best_metrics.get('best_val_dice', best_metrics.get('best_test_dice'))
            prediction_summary = (
                f"Best epoch: {best_metrics['best_epoch']} | "
                f"Train Dice@best val: {best_metrics['best_train_dice']:.4f} | "
                f"Best Val Dice: {best_val:.4f}"
            )
    else:
        (model, train_losses, train_dices, val_losses, val_dices,
         training_time, best_epoch, best_train_dice, best_val_dice) = train(
            model_cls(), train_loader, val_loader,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            use_scheduler=use_scheduler,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
        )
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        torch.save(model.state_dict(), model_path)
        torch.save(
            {
                'best_epoch': best_epoch,
                'best_train_dice': float(best_train_dice),
                'best_val_dice': float(best_val_dice),
                # Backward compatibility with old metrics readers.
                'best_test_dice': float(best_val_dice),
            },
            metrics_path,
        )
        prediction_summary = (
            f"Best epoch: {best_epoch} | "
            f"Train Dice@best val: {best_train_dice:.4f} | "
            f"Best Val Dice: {best_val_dice:.4f}"
        )

        learning_curve_plot(title, train_losses, val_losses, train_dices, val_dices,
                batch_size=batch_size, learning_rate=lr,
                    training_time_seconds=training_time,
                    split_name=curve_split_name)
        print(f'Total training time: {training_time:.2f} seconds')

    # Per-image Dice on held-out evaluation split (test if provided, else validation)
    model.eval()
    per_image_dices = []
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            preds = (model(images) > 0).float()
            for i in range(images.size(0)):
                per_image_dices.append(
                    dice_score(labels[i:i+1].cpu(), preds[i:i+1].cpu())
                )

    num_low_dice = sum(d < 0.80 for d in per_image_dices)
    print(f'Number of {eval_split_name} images with Dice score < 0.80: {num_low_dice} out of {len(per_image_dices)}')
    median_dice = sorted(per_image_dices)[len(per_image_dices) // 2]
    print(f'Median Dice score in {eval_split_name} set: {median_dice:.4f}')

    show_predictions(model, eval_loader, device, title, summary_text=prediction_summary)
    return model

def dice_score(y_true, y_pred):
    """Mean Dice over the batch for binary masks."""

    y_pred = y_pred.bool()
    y_true = y_true.bool()

    intersection = (y_true & y_pred).sum(dim=(1, 2, 3))
    union = y_true.sum(dim=(1, 2, 3)) + y_pred.sum(dim=(1, 2, 3))
    dice_scores = (2.0 * intersection) / (union + 1e-8)  # small epsilon to avoid divide-by-zero

    return dice_scores.mean().item()

class WARWICKDataset(Dataset):
    """Loads the Warwick image/label pairs with some optional preprocessing and augmentation."""

    def __init__(self, root_dir, transform=None, image_transform=None, label_transform=None,
                 autocontrast_green=False, edge_map=False, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.autocontrast_green = autocontrast_green
        self.edge_map = edge_map
        self.augment = augment
        self.images = sorted([f for f in os.listdir(root_dir) if f.startswith('image_')])
        self.labels = sorted([f for f in os.listdir(root_dir) if f.startswith('label_')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        label_path = os.path.join(self.root_dir, self.labels[idx])

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # mask is a single grayscale channel

        image_np = np.array(image)

        if self.autocontrast_green:
            image_np[:, :, 1] = np.array(
                ImageOps.autocontrast(Image.fromarray(image_np[:, :, 1]), 40)
            )

        if self.edge_map:
            # replace blue channel with a Canny edge map made from the red channel
            red = image_np[:, :, 0]
            blurred = cv2.GaussianBlur(red, (7, 7), 10.0)
            otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edges = cv2.Canny(blurred, otsu_thresh * 0.9, otsu_thresh)
            image_np[:, :, 2] = edges
        else:
            image_np[:, :, 2] = 0

        image = Image.fromarray(image_np)

        if self.image_transform:
            image = self.image_transform(image)
            if not self.edge_map:
                image = image[:2]  # drop the empty blue channel so model only gets 2 channels
        if self.label_transform:
            label = self.label_transform(label)

        if self.transform and (self.image_transform is None and self.label_transform is None):
            image = self.transform(image)
            label = self.transform(label)

        if self.augment:
            image, label = self._augment(image, label)

        label = (label > 0).float()  # turn instance ids into 0/1
        return image, label

    def _augment(self, image, label):
        """Random flips, rotation and translation applied to both image and mask."""
        if random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        if random.random() < 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        angle = random.uniform(-180.0, 180.0)
        # use nearest for the label so values stay 0/1 after rotating
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
        label = TF.rotate(label, angle, interpolation=InterpolationMode.NEAREST, fill=0.0)

        h, w = image.shape[-2], image.shape[-1]
        max_dx = int(0.1 * w)
        max_dy = int(0.1 * h)
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        image = TF.affine(image, angle=0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0],
                          interpolation=InterpolationMode.BILINEAR, fill=0.0)
        label = TF.affine(label, angle=0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0],
                          interpolation=InterpolationMode.NEAREST, fill=0.0)

        return image, label

def prepare_dataloader(batch_size, root_dir='WARWICK', image_size=(128, 128),
                       autocontrast_green=False, edge_map=False, augment=False):
    """Build train/test dataloaders for the WARWICK dataset."""
    train_dir = os.path.join(root_dir, 'Train')
    stats_path = os.path.join(root_dir, 'train_image_stats.pt')
    mean, std = get_image_stats(train_dir, image_size, stats_path)

    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])
    label_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    train_dataset = WARWICKDataset(train_dir,
                                   image_transform=image_transform,
                                   label_transform=label_transform,
                                   autocontrast_green=autocontrast_green, edge_map=edge_map,
                                   augment=augment)
    test_dataset = WARWICKDataset(os.path.join(root_dir, 'Test'),
                                  image_transform=image_transform,
                                  label_transform=label_transform,
                                  autocontrast_green=autocontrast_green, edge_map=edge_map,
                                  augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def prepare_dataloader_with_val(batch_size, root_dir='WARWICK', image_size=(128, 128),
                                autocontrast_green=False, edge_map=False, augment=False,
                                val_ratio=0.3, split_seed=42):
    """Same as prepare_dataloader but also splits a validation set out of Train."""

    train_dir = os.path.join(root_dir, 'Train')
    stats_path = os.path.join(root_dir, 'train_image_stats.pt')
    mean, std = get_image_stats(train_dir, image_size, stats_path)

    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])
    label_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # two copies of the same train dir: one with augmentation for training, one without for val
    train_aug_dataset = WARWICKDataset(
        train_dir,
        image_transform=image_transform,
        label_transform=label_transform,
        autocontrast_green=autocontrast_green,
        edge_map=edge_map,
        augment=augment,
    )
    train_no_aug_dataset = WARWICKDataset(
        train_dir,
        image_transform=image_transform,
        label_transform=label_transform,
        autocontrast_green=autocontrast_green,
        edge_map=edge_map,
        augment=False,
    )

    num_samples = len(train_aug_dataset)
    indices = list(range(num_samples))
    rng = random.Random(split_seed)
    rng.shuffle(indices)

    val_size = max(1, int(num_samples * val_ratio))
    val_size = min(val_size, num_samples - 1)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(train_aug_dataset, train_indices)
    val_subset = Subset(train_no_aug_dataset, val_indices)

    test_dataset = WARWICKDataset(
        os.path.join(root_dir, 'Test'),
        image_transform=image_transform,
        label_transform=label_transform,
        autocontrast_green=autocontrast_green,
        edge_map=edge_map,
        augment=False,
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Total Train samples: {num_samples}')
    print(f'Total Test samples: {len(test_dataset)}')
    print(f'Train/Val split: {len(train_subset)} / {len(val_subset)} ({(1.0 - val_ratio):.0%}/{val_ratio:.0%})')

    return train_loader, val_loader, test_loader

def compute_mean_std(root_dir, image_size):
    """Compute per-channel mean and std of the images so we can normalize later."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    dataset = WARWICKDataset(root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sum_sq += (images ** 2).sum(dim=(0, 2, 3))
        total_pixels += batch_pixels

    mean = channel_sum / total_pixels
    variance = (channel_sum_sq / total_pixels) - (mean ** 2)
    variance = torch.clamp(variance, min=0.0)  # floating point can push this slightly negative
    std = torch.sqrt(variance)

    mean = mean.detach().float().cpu()
    std = std.detach().float().cpu()
    mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = torch.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6)
    std = torch.clamp(std, min=1e-6)

    print(f"Computed mean: {mean}, std: {std} from data at {root_dir}")

    return mean, std

def get_image_stats(train_dir, image_size, stats_path):
    """Load cached mean/std from disk, otherwise compute them and cache."""
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location='cpu')
        mean = stats['mean'].detach().float().cpu()
        std = stats['std'].detach().float().cpu()

        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = torch.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6)
        std = torch.clamp(std, min=1e-6)

        # recompute if the cache is broken (e.g. std ended up as zero)
        if not torch.all(std > 1e-6):
            mean, std = compute_mean_std(train_dir, image_size)
            torch.save({'mean': mean, 'std': std}, stats_path)

        return mean, std

    mean, std = compute_mean_std(train_dir, image_size)
    torch.save({'mean': mean, 'std': std}, stats_path)
    return mean, std

def learning_curve_plot(
    title,
    train_losses,
    test_losses,
    train_dices,
    test_dices,
    batch_size,
    learning_rate,
    training_time_seconds,
    split_name="test",
):
    """Plot loss and Dice curves side-by-side and save the figure to results/."""
    fsizes = [13, 10, 8]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, y=1.1, fontsize=fsizes[0])

    formatted_time = f"{int(training_time_seconds // 3600)}h {int((training_time_seconds % 3600) // 60)}min {int(training_time_seconds % 60)}s"
    sub = f"| Batch size:{batch_size} | Learning rate:{learning_rate} | "
    sub += f"Number of Epochs:{len(train_losses)} | Training time:{formatted_time} |"
    fig.text(0.5, 0.99, sub, ha="center", fontsize=fsizes[1])

    x = range(1, len(train_losses) + 1)

    axs[0].plot(x, train_losses, label=f"Final train loss: {train_losses[-1]:.4f}")
    axs[0].plot(x, test_losses, label=f"Final {split_name} loss: {test_losses[-1]:.4f}")
    axs[0].set_title("Losses", fontsize=fsizes[1])
    axs[0].set_xlabel("Epoch", fontsize=fsizes[1])
    axs[0].set_ylabel("Loss", fontsize=fsizes[1])
    axs[0].legend(fontsize=fsizes[2])
    axs[0].tick_params(axis="both", labelsize=fsizes[2])
    axs[0].grid()

    axs[1].plot(x, train_dices, label=f"Final train dice: {train_dices[-1]:.4f}")
    axs[1].plot(x, test_dices, label=f"Final {split_name} dice: {test_dices[-1]:.4f}")
    axs[1].set_title("Dice Score", fontsize=fsizes[1])
    axs[1].set_xlabel("Epoch", fontsize=fsizes[1])
    axs[1].set_ylabel("Dice", fontsize=fsizes[2])
    axs[1].legend(fontsize=fsizes[2])
    axs[1].tick_params(axis="both", labelsize=fsizes[2])
    axs[1].grid()

    plt.savefig(f"results/{title}_learning_curve.png", bbox_inches="tight")
    plt.show()

def show_predictions(model, test_loader, device, title, summary_text=None):
    """Find the best and worst predictions by Dice and plot them next to the ground truth."""
    model.eval()
    dataset = test_loader.dataset
    scores = []  # each entry: (dice, index, image, label, pred)

    # grab the normalize stats so we can undo them when displaying the image
    norm_mean = None
    norm_std = None
    image_transform = getattr(dataset, 'image_transform', None)
    if image_transform is not None and hasattr(image_transform, 'transforms'):
        for t in image_transform.transforms:
            if isinstance(t, transforms.Normalize):
                norm_mean = torch.tensor(t.mean).view(-1, 1, 1)
                norm_std = torch.tensor(t.std).view(-1, 1, 1)
                break

    with torch.no_grad():
        idx = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = (logits > 0).float()

            for i in range(images.size(0)):
                d = dice_score(labels[i:i+1].cpu(), preds[i:i+1].cpu())
                scores.append((d, idx, images[i].cpu(), labels[i].cpu(), preds[i].cpu()))
                idx += 1

    scores.sort(key=lambda x: x[0])
    worst = scores[0]
    best = scores[-1]

    fig, axs = plt.subplots(2, 3, figsize=(9, 7))
    if summary_text:
        fig.suptitle(f"{title}\n{summary_text}")
    else:
        fig.suptitle(title)

    for row, (tag, sample) in enumerate([("Best", best), ("Worst", worst)]):
        d, sample_idx, img, gt, pred = sample
        label_name = dataset.labels[sample_idx]

        img_vis = img.clone()
        if norm_mean is not None and norm_std is not None:
            if img_vis.size(0) == norm_mean.size(0):
                img_vis = img_vis * norm_std + norm_mean
            elif img_vis.size(0) == 2 and norm_mean.size(0) >= 2:
                # 2-channel image but stats are 3-channel, just use the first two
                img_vis = img_vis * norm_std[:2] + norm_mean[:2]
        img_vis = img_vis.clamp(0.0, 1.0)

        # pad out to 3 channels so matplotlib can show it as RGB
        if img_vis.size(0) == 1:
            img_rgb = img_vis.repeat(3, 1, 1)
        elif img_vis.size(0) == 2:
            img_rgb = torch.cat([img_vis, torch.zeros_like(img_vis[:1])], dim=0)
        else:
            img_rgb = img_vis[:3]

        img_rgb = img_rgb.permute(1, 2, 0).numpy()

        axs[row, 0].imshow(img_rgb)
        axs[row, 0].set_title(f"{tag} — {label_name} (Dice {d:.3f})\nOriginal image")
        axs[row, 1].imshow(pred.squeeze().numpy(), cmap="gray")
        axs[row, 1].set_title("Generated mask")
        axs[row, 2].imshow(gt.squeeze().numpy(), cmap="gray")
        axs[row, 2].set_title("Ground truth mask")
        for ax in axs[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"results/{title}_best_worst.png", bbox_inches="tight")
    plt.show()

