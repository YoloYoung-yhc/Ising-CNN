"""Provide simulation and visualization utilities for the 2D Ising model."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def metropolis(
    spins: np.ndarray,
    temperature: float,
    steps: int = 25_000,
    coupling: float = 1.0,
    boltzmann_constant: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evolve a square Ising lattice with the Metropolis algorithm.

    Periodic boundary conditions are applied. The input array is updated in
    place during the simulation.

    Args:
        spins: Square array containing spin values of -1 and 1.
        temperature: Simulation temperature. Must be positive.
        steps: Number of Monte Carlo updates to perform.
        coupling: Nearest-neighbor interaction strength.
        boltzmann_constant: Boltzmann constant used in the acceptance rule.
        seed: Optional seed for the random-number generator.

    Returns:
        A tuple containing the energy history, magnetization history, and the
        final spin configuration.

    Raises:
        ValueError: If ``spins`` is not a square two-dimensional array or if
            ``temperature`` is not positive.
    """
    if spins.ndim != 2 or spins.shape[0] != spins.shape[1]:
        raise ValueError("spins must be a square two-dimensional array")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    lattice_size = spins.shape[0]
    rng = np.random.default_rng(seed)
    energy_history = np.empty(steps, dtype=float)
    magnetization_history = np.empty(steps, dtype=float)
    energy = -coupling * np.sum(
        spins * (np.roll(spins, -1, axis=0) + np.roll(spins, -1, axis=1))
    )
    total_spin = spins.sum()

    for step in range(steps):
        row = rng.integers(lattice_size)
        column = rng.integers(lattice_size)
        neighbor_sum = (
            spins[(row - 1) % lattice_size, column]
            + spins[(row + 1) % lattice_size, column]
            + spins[row, (column - 1) % lattice_size]
            + spins[row, (column + 1) % lattice_size]
        )
        energy_change = 2 * coupling * spins[row, column] * neighbor_sum

        if energy_change <= 0 or rng.random() < np.exp(
            -energy_change / (boltzmann_constant * temperature)
        ):
            spins[row, column] *= -1
            energy += energy_change
            total_spin += 2 * spins[row, column]

        energy_history[step] = energy
        magnetization_history[step] = total_spin / spins.size

    return energy_history, magnetization_history, spins


def plot_energy(energy_history: np.ndarray, temperature: float) -> None:
    """Plot the energy history for one simulation temperature.

    Args:
        energy_history: Energy value recorded at each Monte Carlo step.
        temperature: Temperature used for the simulation.
    """
    plt.plot(energy_history, label=f"T={temperature:g}")
    plt.xlabel("Monte Carlo step")
    plt.ylabel("Energy")
    plt.legend()


def plot_magnetization(magnetization_history: np.ndarray, temperature: float) -> None:
    """Plot the magnetization history for one simulation temperature.

    Args:
        magnetization_history: Magnetization recorded at each Monte Carlo step.
        temperature: Temperature used for the simulation.
    """
    plt.plot(magnetization_history, label=f"T={temperature:g}")
    plt.xlabel("Monte Carlo step")
    plt.ylabel("Magnetization")
    plt.legend()


def plot_spin_configuration(spins: np.ndarray) -> None:
    """Display an Ising spin configuration.

    Args:
        spins: Two-dimensional array containing the spin configuration.
    """
    plt.imshow(
        spins, cmap=ListedColormap(["white", "steelblue"]), interpolation="nearest", vmin=-1, vmax=1
    )
    plt.axis("off")


def save_pickle(data: object, output_path: str | Path) -> None:
    """Serialize an object to a pickle file.

    Parent directories are created when needed.

    Args:
        data: Python object to serialize.
        output_path: Destination path for the pickle file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(data, file)
