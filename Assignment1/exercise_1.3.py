from models import ResUNetV2
from utils import prepare_dataloader, run_experiment


if __name__ == "__main__":
    batch_size = 16
    train_loader, test_loader = prepare_dataloader(batch_size, autocontrast_green=True, edge_map=False)
    run_experiment(
        ResUNetV2, train_loader, test_loader,
        title="ResUNet",
        model_path='results/models/ResUNet.pth',
        batch_size=batch_size,
        # weight_decay=1e-4,
    )
