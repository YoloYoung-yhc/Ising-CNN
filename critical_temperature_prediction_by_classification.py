"""Estimate the Ising critical temperature with a phase-classification CNN."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ising import metropolis
from ising_cnn import IsingClassifier

EXACT_CRITICAL_TEMPERATURE = 2.269


def main() -> None:
    """Estimate the transition from phase probabilities across temperature.

    The predicted critical temperature is the point closest to equal low- and
    high-temperature phase probabilities.
    """
    temperatures = np.linspace(1.5, 3.0, 150)
    lattice_size, monte_carlo_steps, samples_per_temperature = 20, 25_000, 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IsingClassifier().to(device)
    model.load_state_dict(torch.load("models/ising_classifier.pth", map_location=device, weights_only=True))
    model.eval()
    low_temperature_probabilities = []

    with torch.no_grad():
        for temperature in temperatures:
            low_temperature_predictions = 0
            for _ in range(samples_per_temperature):
                spins = np.ones((lattice_size, lattice_size), dtype=np.int8)
                _, _, spins = metropolis(spins, temperature, steps=monte_carlo_steps)
                inputs = torch.tensor(spins, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                low_temperature_predictions += (model(inputs).argmax(dim=1).item() == 1)
            low_temperature_probabilities.append(low_temperature_predictions / samples_per_temperature)

    low_temperature_probabilities = np.asarray(low_temperature_probabilities)
    estimated_temperature = temperatures[np.abs(low_temperature_probabilities - 0.5).argmin()]
    relative_error = abs(estimated_temperature - EXACT_CRITICAL_TEMPERATURE) / EXACT_CRITICAL_TEMPERATURE
    print(f"Estimated critical temperature: {estimated_temperature:.3f}")
    print(f"Relative error: {relative_error:.2%}")

    Path("results").mkdir(exist_ok=True)
    plt.plot(temperatures, low_temperature_probabilities, label="Low-temperature probability")
    plt.plot(temperatures, 1 - low_temperature_probabilities, label="High-temperature probability")
    plt.axvline(estimated_temperature, color="red", linestyle="--", label=f"Estimated Tc = {estimated_temperature:.3f}")
    plt.xlabel("Temperature")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/critical_temperature_classification.png", dpi=300)


if __name__ == "__main__":
    main()
