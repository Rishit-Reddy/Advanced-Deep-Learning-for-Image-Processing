import os
from helper_resnet18 import train_model

def log_experiment(exp_name, lr, wd, bs, train_loss, train_acc, best_loss, best_acc, best_auc):
    summary_path = "resnet18_experiments_summary.md"
    header_needed = not os.path.exists(summary_path)
    with open(summary_path, "a") as f:
        if header_needed:
            f.write("# ResNet18 Hyperparameter Sweep Results\n\n")
            f.write("| Trial | LR | Weight Decay | Batch | Train Loss | Train Acc | Best Val Loss | Best Val Acc | Best Val AUC |\n")
            f.write("|-------|----|--------------|-------|------------|-----------|---------------|--------------|--------------|\n")
        f.write(
            f"| {exp_name} | {lr:.2e} | {wd:.2e} | {bs} | {train_loss:.4f} | {train_acc:.4f} | "
            f"{best_loss:.4f} | {best_acc:.4f} | {best_auc:.4f} |\n"
        )

def objective(trial):
    # Optuna picks hyperparameters
    lr = trial.suggest_float("lr", 5e-6, 5e-4, log=True)
    wd = trial.suggest_float("wd", 1e-7, 1e-4, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32, 64])

    exp_name = f"resnet18_trial_{trial.number}_bs{bs}"

    # Run the model
    metrics = train_model(
        lr=lr,
        weight_decay=wd,
        batch_size=bs,
        samples_per_epoch=25000, # Increased for more data variety per trial
        epochs=12,
        exp_name=exp_name
    )

    # Log to our markdown file
    log_experiment(
        exp_name, lr, wd, bs,
        metrics["train_loss"], metrics["train_acc"],
        metrics["best_loss"], metrics["best_acc"], metrics["best_auc"],
    )

    return metrics["best_auc"]

if __name__ == "__main__":
    # Optuna sweep is parked — single fixed-config run while we validate the
    # regularization bundle. Re-enable by importing optuna and calling
    # study.optimize(objective, ...) using the helpers above.
    lr = 3e-5
    wd = 1e-5
    bs = 32
    exp_name = f"resnet18_fixed_lr{lr:.0e}_wd{wd:.0e}_bs{bs}"

    metrics = train_model(
        lr=lr,
        weight_decay=wd,
        batch_size=bs,
        samples_per_epoch=25000,
        exp_name=exp_name,
    )

    log_experiment(
        exp_name, lr, wd, bs,
        metrics["train_loss"], metrics["train_acc"],
        metrics["best_loss"], metrics["best_acc"], metrics["best_auc"],
    )

    print(
        f"\nRun complete. Train Loss {metrics['train_loss']:.4f}, "
        f"Train Acc {metrics['train_acc']:.4f}, Val Loss {metrics['best_loss']:.4f}, "
        f"Val Acc {metrics['best_acc']:.4f}, Val AUC {metrics['best_auc']:.4f}"
    )
