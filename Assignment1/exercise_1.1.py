from models import EncoderDecoderModelV2
from utils import prepare_dataloader, run_experiment


if __name__ == "__main__":
    batch_size = 16
    train_loader, test_loader = prepare_dataloader(batch_size, autocontrast_green=False, edge_map=False)
    run_experiment(
        EncoderDecoderModelV2, train_loader, test_loader,
        title="EncoderDecoder",
        model_path='results/models/EncoderDecoder.pth',
        batch_size=batch_size,
    )
