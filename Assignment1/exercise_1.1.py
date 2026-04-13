import os
import time
import torch
import torch.nn as nn
from models import EncoderDecoderModel
from utils import dice_score, prepare_dataloader, learning_curve_plot, show_predictions


def train(train_loader, test_loader, epochs=50, lr=1e-3, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EncoderDecoderModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    train_dices, test_dices = [], []
    train_accuracy, test_accuracy = [], []

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

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        test_dice = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                test_dice += dice_score(labels.cpu(), (outputs.cpu() > 0).float()) * images.size(0)

        test_losses.append(test_loss / len(test_loader.dataset))
        test_dices.append(test_dice / len(test_loader.dataset))

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Train Dice: {train_dices[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Dice: {test_dices[-1]:.4f}')

    training_time = time.time() - starttime

    return model, train_losses, train_dices, test_losses, test_dices, training_time

if __name__ == "__main__":
    batch_size = 16
    lr = 1e-3
    title = "1.1 : Without Skip Connections(16)"
    model_path = 'encoder_decoder_model.pth'

    train_loader, test_loader = prepare_dataloader(batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(model_path):
        print(f'Loading existing model from {model_path}, skipping training.')
        model = EncoderDecoderModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model, train_losses, train_dices, test_losses, test_dices, training_time = train(
            train_loader, test_loader, epochs=50, lr=lr, batch_size=batch_size
        )
        torch.save(model.state_dict(), model_path)
        learning_curve_plot(title, train_losses, test_losses, train_dices, test_dices,
                            batch_size=batch_size, learning_rate=lr,
                            training_time_seconds=training_time)
        print(f'Total training time: {training_time:.2f} seconds')

    show_predictions(model, test_loader, device, title)