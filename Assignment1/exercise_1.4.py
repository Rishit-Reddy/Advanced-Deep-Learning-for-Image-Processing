from models import UNetV2, UNetV3
from utils import prepare_dataloader_with_val, run_experiment


if __name__ == "__main__":
    batch_size = 16
    lr = 2e-3

    # 1.4a: augmentation only
    train_loader, val_loader, test_loader = prepare_dataloader_with_val(
        batch_size,
        autocontrast_green=False,
        edge_map=False,
        augment=True,
        val_ratio=0.3,
    )
    run_experiment(
        UNetV2,
        train_loader,
        val_loader,
        title="UNet_AugOnly",
        model_path="results/models/UNet_AugOnly.pth",
        batch_size=batch_size,
        lr=lr,
        test_loader=test_loader,
        curve_split_name="validation",
    )

    # 1.4b: dropout only
    train_loader, val_loader, test_loader = prepare_dataloader_with_val(
        batch_size,
        autocontrast_green=False,
        edge_map=False,
        augment=False,
        val_ratio=0.3,
    )
    run_experiment(
        UNetV3,
        train_loader,
        val_loader,
        title="UNet_DropoutOnly",
        model_path="results/models/UNet_DropoutOnly.pth",
        batch_size=batch_size,
        lr=lr,
        test_loader=test_loader,
        curve_split_name="validation",
    )

    # 1.4c: learning-rate scheduler only
    train_loader, val_loader, test_loader = prepare_dataloader_with_val(
        batch_size,
        autocontrast_green=False,
        edge_map=False,
        augment=False,
        val_ratio=0.3,
    )
    run_experiment(
        UNetV2,
        train_loader,
        val_loader,
        title="UNet_SchedulerOnly",
        model_path="results/models/UNet_SchedulerOnly.pth",
        batch_size=batch_size,
        lr=lr,
        test_loader=test_loader,
        curve_split_name="validation",
        use_scheduler=True,
    )

    # 1.4d: all three techniques combined
    train_loader, val_loader, test_loader = prepare_dataloader_with_val(
        batch_size,
        autocontrast_green=False,
        edge_map=False,
        augment=True,
        val_ratio=0.3,
    )
    run_experiment(
        UNetV3,
        train_loader,
        val_loader,
        title="UNet_AllTechniques",
        model_path="results/models/UNet_AllTechniques.pth",
        batch_size=batch_size,
        lr=lr,
        test_loader=test_loader,
        curve_split_name="validation",
        use_scheduler=True,
    )
