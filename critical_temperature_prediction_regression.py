"""Estimate the Ising critical temperature with a magnetization-regression CNN."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ising import metropolis
from ising_cnn import IsingRegressor

EXACT_CRITICAL_TEMPERATURE = 2.269


def main() -> None:
    """Estimate the transition from the predicted magnetization gradient.

    The predicted critical temperature is the location of the steepest
    magnetization decrease.
    """
    temperatures = np.linspace(1.5, 3.0, 150)
    lattice_size, monte_carlo_steps = 20, 25_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IsingRegressor().to(device)
    model.load_state_dict(torch.load("models/ising_regressor.pth", map_location=device, weights_only=True))
    model.eval()
    predicted_magnetizations = []

    with torch.no_grad():
        for temperature in temperatures:
            spins = np.ones((lattice_size, lattice_size), dtype=np.int8)
            _, _, spins = metropolis(spins, temperature, steps=monte_carlo_steps)
            inputs = torch.tensor(spins, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            predicted_magnetizations.append(model(inputs).item())

    magnetization_gradient = np.gradient(predicted_magnetizations, temperatures)
    estimated_temperature = temperatures[magnetization_gradient.argmin()]
    relative_error = abs(estimated_temperature - EXACT_CRITICAL_TEMPERATURE) / EXACT_CRITICAL_TEMPERATURE
    print(f"Estimated critical temperature: {estimated_temperature:.3f}")
    print(f"Relative error: {relative_error:.2%}")

    Path("results").mkdir(exist_ok=True)
    plt.plot(temperatures, predicted_magnetizations, "o", markersize=3, label="Predicted magnetization")
    plt.axvline(estimated_temperature, color="red", linestyle="--", label=f"Estimated Tc = {estimated_temperature:.3f}")
    plt.xlabel("Temperature")
    plt.ylabel("Absolute magnetization")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/critical_temperature_regression.png", dpi=300)


if __name__ == "__main__":
    main()
