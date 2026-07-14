"""Train a CNN to classify Ising configurations by temperature phase."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import IsingDataset
from ising_cnn import IsingClassifier

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01


def evaluate(model: nn.Module, data_loader: DataLoader, loss_function: nn.Module, device: torch.device) -> tuple[float, dict[str, float]]:
    """Evaluate a classifier and calculate binary-classification metrics.

    Args:
        model: Classification model to evaluate.
        data_loader: Data loader containing evaluation samples.
        loss_function: Loss function used to calculate mean loss.
        device: Device on which evaluation is performed.

    Returns:
        A tuple containing the mean loss and a dictionary with accuracy,
        precision, recall, and F1 score.
    """
    model.eval()
    total_loss = true_positive = true_negative = false_positive = false_negative = 0
    with torch.no_grad():
        for spins, targets in data_loader:
            logits = model(spins.to(device))
            targets = targets.to(device)
            total_loss += loss_function(logits, targets).item()
            predictions = logits.argmax(dim=1)
            true_positive += ((predictions == 1) & (targets == 1)).sum().item()
            true_negative += ((predictions == 0) & (targets == 0)).sum().item()
            false_positive += ((predictions == 1) & (targets == 0)).sum().item()
            false_negative += ((predictions == 0) & (targets == 1)).sum().item()

    total = true_positive + true_negative + false_positive + false_negative
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    return total_loss / len(data_loader), {
        "accuracy": (true_positive + true_negative) / total,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-8),
    }


def main() -> None:
    """Train the classifier and save its weights and learning curves."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(IsingDataset("data/ising_configs_labels_train.pkl", torch.long), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(IsingDataset("data/ising_configs_labels_test.pkl", torch.long), batch_size=BATCH_SIZE)
    model = IsingClassifier().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    train_losses, test_losses = [], []
    metrics = {name: [] for name in ("accuracy", "precision", "recall", "f1")}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for spins, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(spins.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        test_loss, epoch_metrics = evaluate(model, test_loader, loss_function, device)
        test_losses.append(test_loss)
        for name, value in epoch_metrics.items():
            metrics[name].append(value)
        print(f"Epoch {epoch}/{EPOCHS} | test loss: {test_loss:.3f} | accuracy: {epoch_metrics['accuracy']:.2%}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/ising_classifier.pth")
    _plot_results(train_losses, test_losses, metrics)


def _plot_results(train_losses: list[float], test_losses: list[float], metrics: dict[str, list[float]]) -> None:
    """Save loss and classification-metric plots.

    Args:
        train_losses: Mean training loss for each epoch.
        test_losses: Mean test loss for each epoch.
        metrics: Per-epoch classification metrics keyed by metric name.
    """
    Path("results").mkdir(exist_ok=True)
    epochs = range(1, EPOCHS + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, test_losses, label="Test")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Classification loss")
    axes[0].legend()
    for name, values in metrics.items():
        axes[1].plot(epochs, values, label=name.title())
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Classification metrics")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig("results/classification_training.png", dpi=300)


if __name__ == "__main__":
    main()
