"""Run baseline Monte Carlo simulations and save physical-observable plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ising import metropolis, plot_energy, plot_magnetization, plot_spin_configuration


def main() -> None:
    """Simulate selected configurations and temperature-dependent observables."""
    lattice_size, monte_carlo_steps = 20, 25_000
    initial_spins = np.ones((lattice_size, lattice_size), dtype=np.int8)
    Path("results").mkdir(exist_ok=True)

    for temperature in (0.01, 2.27, 5.0):
        energy_history, magnetization_history, spins = metropolis(initial_spins.copy(), temperature, steps=monte_carlo_steps)
        print(f"T={temperature:g} | final energy: {energy_history[-1]:.1f} | final magnetization: {magnetization_history[-1]:.3f}")
        plt.figure(figsize=(4, 4))
        plot_spin_configuration(spins)
        plt.tight_layout()
        plt.savefig(f"results/spin_configuration_T_{temperature:g}.png", dpi=300)
        plt.close()

    temperatures = np.arange(0.2, 5.2, 0.01)
    energies, heat_capacities, magnetizations, susceptibilities = [], [], [], []
    for temperature in temperatures:
        energy_history, magnetization_history, _ = metropolis(initial_spins.copy(), temperature, steps=monte_carlo_steps)
        equilibrium_slice = slice(monte_carlo_steps // 2, None)
        energy_per_spin = energy_history[equilibrium_slice] / initial_spins.size
        equilibrium_magnetization = magnetization_history[equilibrium_slice]
        energies.append(energy_per_spin.mean())
        heat_capacities.append(energy_per_spin.var() / temperature**2)
        magnetizations.append(np.abs(equilibrium_magnetization).mean())
        susceptibilities.append(equilibrium_magnetization.var())

    _plot_observables(temperatures, energies, heat_capacities, magnetizations, susceptibilities)


def _plot_observables(temperatures: np.ndarray, energies: list[float], heat_capacities: list[float], magnetizations: list[float], susceptibilities: list[float]) -> None:
    """Save a four-panel summary of thermodynamic observables.

    Args:
        temperatures: Temperatures at which observables were calculated.
        energies: Mean energy per spin at each temperature.
        heat_capacities: Heat capacity at each temperature.
        magnetizations: Mean absolute magnetization at each temperature.
        susceptibilities: Magnetic susceptibility at each temperature.
    """
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, values, title, ylabel in zip(
        axes.flat,
        (energies, heat_capacities, magnetizations, susceptibilities),
        ("Energy", "Heat capacity", "Magnetization", "Magnetic susceptibility"),
        ("Energy per spin", "Heat capacity", "Absolute magnetization", "Susceptibility"),
    ):
        axis.plot(temperatures, values, 'o')
        axis.set(xlabel="Temperature", ylabel=ylabel, title=title)
    figure.tight_layout()
    figure.savefig("results/thermodynamic_observables.png", dpi=300)


if __name__ == "__main__":
    main()
