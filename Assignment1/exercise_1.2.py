from models import UNetV2
from utils import prepare_dataloader, run_experiment


if __name__ == "__main__":
    batch_size = 16
    train_loader, test_loader = prepare_dataloader(batch_size)
    run_experiment(
        UNetV2, train_loader, test_loader,
        title="UNetV2",
        model_path='results/models/UNetV2.pth',
        batch_size=batch_size,
    )
