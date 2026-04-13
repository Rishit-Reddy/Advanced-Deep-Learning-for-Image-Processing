
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os

def dice_score(y_true, y_pred):
    '''
    Computes Dice score for an list of images(batch safe).
    TODO: Remember binarize predictions, cond y_pred > 0.5 before passing to this function.

    Args:
        y_true: torch.Tensor of shape (N, H, W) with binary values (0 or 1)
        y_pred: torch.Tensor of shape (N, H, W) with binary values (0 or 1)
    Returns:
        dice_score: float, the average Dice score across the batch
    '''

    y_pred = y_pred.bool()
    y_true = y_true.bool()

    intersection = (y_true & y_pred).sum(dim=(1, 2, 3)) # keeps batch dimension
    union = y_true.sum(dim=(1, 2, 3)) + y_pred.sum(dim=(1, 2, 3)) # keeps batch dimension
    dice_scores = (2.0 * intersection) / (union + 1e-8) # keeps batch dimension
    
    return dice_scores.mean().item()

class WARWICKDataset(Dataset):
    '''
    Custom Dataset class for loading Warwick data.
    '''

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(root_dir) if f.startswith('image_')])
        self.labels = sorted([f for f in os.listdir(root_dir) if f.startswith('label_')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        label_path = os.path.join(self.root_dir, self.labels[idx])
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # grayscale binary mask
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        image = image[:2]  # drop blue channel (always 0), keep R+G
        label = (label > 0).float()  # binarize: instance ids -> {0, 1}
        return image, label

def prepare_dataloader(batch_size, root_dir='WARWICK', image_size=(128, 128)):
    '''
        Load WARWICK dataset and prepare dataloaders for training and testing.

        Args:
            batch_size: int, batch size for both loaders
            root_dir: str, path to the WARWICK folder containing Train/ and Test/
            image_size: tuple, (H, W) to resize images and masks to
        Returns:
            train_loader, test_loader
    '''
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    train_dataset = WARWICKDataset(os.path.join(root_dir, 'Train'), transform=transform)
    test_dataset = WARWICKDataset(os.path.join(root_dir, 'Test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader




def learning_curve_plot(
    title,
    train_losses,
    test_losses,
    train_dices,
    test_dices,
    batch_size,
    learning_rate,
    training_time_seconds,
):
    fsizes = [13, 10, 8]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, y=1.1, fontsize=fsizes[0])

    formatted_time = f"{int(training_time_seconds // 3600)}h {int((training_time_seconds % 3600) // 60)}min {int(training_time_seconds % 60)}s"
    sub = f"| Batch size:{batch_size} | Learning rate:{learning_rate} | "
    sub += f"Number of Epochs:{len(train_losses)} | Training time:{formatted_time} |"
    fig.text(0.5, 0.99, sub, ha="center", fontsize=fsizes[1])

    x = range(1, len(train_losses) + 1)

    axs[0].plot(x, train_losses, label=f"Final train loss: {train_losses[-1]:.4f}")
    axs[0].plot(x, test_losses, label=f"Final test loss: {test_losses[-1]:.4f}")
    axs[0].set_title("Losses", fontsize=fsizes[1])
    axs[0].set_xlabel("Epoch", fontsize=fsizes[1])
    axs[0].set_ylabel("Loss", fontsize=fsizes[1])
    axs[0].legend(fontsize=fsizes[2])
    axs[0].tick_params(axis="both", labelsize=fsizes[2])
    axs[0].grid()

    axs[1].plot(x, train_dices, label=f"Final train dice: {train_dices[-1]:.4f}")
    axs[1].plot(x, test_dices, label=f"Final test dice: {test_dices[-1]:.4f}")
    axs[1].set_title("Dice Score", fontsize=fsizes[1])
    axs[1].set_xlabel("Epoch", fontsize=fsizes[1])
    axs[1].set_ylabel("Dice", fontsize=fsizes[2])
    axs[1].legend(fontsize=fsizes[2])
    axs[1].tick_params(axis="both", labelsize=fsizes[2])
    axs[1].grid()

    plt.savefig(f"results/{title}_learning_curve.png", bbox_inches="tight")
    plt.show()


def show_predictions(model, test_loader, device, title):
    '''
    Runs the model on the test set, finds the single best and worst samples
    by Dice score, and plots input / ground truth / prediction side-by-side.
    '''
    model.eval()
    dataset = test_loader.dataset
    scores = []  # (dice, index, image, label, pred)

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

    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle(title)

    for row, (tag, sample) in enumerate([("Best", best), ("Worst", worst)]):
        d, sample_idx, img, gt, pred = sample
        label_name = dataset.labels[sample_idx]
        # img has 2 channels (R, G); add a zero blue channel back for display
        img_rgb = torch.cat([img, torch.zeros_like(img[:1])], dim=0).permute(1, 2, 0).numpy()

        axs[row, 0].imshow(img_rgb)
        axs[row, 0].set_title(f"{tag} — {label_name} (Dice {d:.3f})")
        axs[row, 1].imshow(gt.squeeze().numpy(), cmap="gray")
        axs[row, 1].set_title("Ground truth")
        axs[row, 2].imshow(pred.squeeze().numpy(), cmap="gray")
        axs[row, 2].set_title("Prediction")
        for ax in axs[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"results/{title}_best_worst.png", bbox_inches="tight")
    plt.show()

