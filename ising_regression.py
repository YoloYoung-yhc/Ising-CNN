"""Train a CNN to predict absolute magnetization from Ising configurations."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import IsingDataset
from ising_cnn import IsingRegressor

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01


def evaluate(model: nn.Module, data_loader: DataLoader, loss_function: nn.Module, device: torch.device) -> tuple[float, float]:
    """Calculate mean squared error and mean absolute error.

    Args:
        model: Regression model to evaluate.
        data_loader: Data loader containing evaluation samples.
        loss_function: Loss function used to calculate mean squared error.
        device: Device on which evaluation is performed.

    Returns:
        A tuple containing the mean squared error and mean absolute error.
    """
    model.eval()
    total_loss = total_absolute_error = sample_count = 0.0
    with torch.no_grad():
        for spins, targets in data_loader:
            predictions = model(spins.to(device)).squeeze(1)
            targets = targets.to(device)
            total_loss += loss_function(predictions, targets).item()
            total_absolute_error += torch.abs(predictions - targets).sum().item()
            sample_count += targets.size(0)
    return total_loss / len(data_loader), total_absolute_error / sample_count


def main() -> None:
    """Train the regressor and save its weights and learning curves."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(IsingDataset("data/ising_configs_magnetization_train.pkl", torch.float32), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(IsingDataset("data/ising_configs_magnetization_test.pkl", torch.float32), batch_size=BATCH_SIZE)
    model = IsingRegressor().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    train_losses, test_losses, maes = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for spins, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(spins.to(device)).squeeze(1), targets.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        test_loss, mae = evaluate(model, test_loader, loss_function, device)
        test_losses.append(test_loss)
        maes.append(mae)
        print(f"Epoch {epoch}/{EPOCHS} | test MSE: {test_loss:.3f} | MAE: {mae:.3f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/ising_regressor.pth")
    _plot_results(train_losses, test_losses, maes)


def _plot_results(train_losses: list[float], test_losses: list[float], maes: list[float]) -> None:
    """Save regression learning curves.

    Args:
        train_losses: Mean training loss for each epoch.
        test_losses: Mean test loss for each epoch.
        maes: Mean absolute error for each epoch.
    """
    Path("results").mkdir(exist_ok=True)
    epochs = range(1, EPOCHS + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, test_losses, label="Test")
    axes[0].set(xlabel="Epoch", ylabel="MSE", title="Regression loss")
    axes[0].legend()
    axes[1].plot(epochs, maes)
    axes[1].set(xlabel="Epoch", ylabel="MAE", title="Mean absolute error")
    figure.tight_layout()
    figure.savefig("results/regression_training.png", dpi=300)


if __name__ == "__main__":
    main()
