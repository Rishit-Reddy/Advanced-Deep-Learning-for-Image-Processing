
import matplotlib.pyplot as plt
import torch

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

    intersection = (y_true & y_pred).sum(dim=(1, 2)) # keeps batch dimension
    union = y_true.sum(dim=(1, 2)) + y_pred.sum(dim=(1, 2)) # keeps batch dimension
    dice_scores = (2.0 * intersection) / (union + 1e-8) # keeps batch dimension
    
    return dice_scores.mean().item()


def learning_curve_plot(
    title,
    train_losses,
    test_losses,
    train_accuracy,
    test_accuracy,
    batch_size,
    learning_rate,
    training_time_seconds,
):
    fsizes = [13, 10, 8]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, y=1.1, fontsize=fsizes[0])

    formatted_time = f"{training_time_seconds // 3600}h {(training_time_seconds % 3600) // 60}min"
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

    axs[1].plot(x, train_accuracy, label=f"Final train accuracy: {train_accuracy[-1]:.4f}%")
    axs[1].plot(x, test_accuracy, label=f"Final test accuracy: {test_accuracy[-1]:.4f}%")
    axs[1].set_title("Accuracy", fontsize=fsizes[1])
    axs[1].set_xlabel("Epoch", fontsize=fsizes[1])
    axs[1].set_ylabel("Accuracy (%)", fontsize=fsizes[2])
    axs[1].legend(fontsize=fsizes[2])
    axs[1].tick_params(axis="both", labelsize=fsizes[2])
    axs[1].grid()

