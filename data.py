"""Provide dataset generation and loading utilities for Ising CNN experiments."""

from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from ising import metropolis, save_pickle

CRITICAL_TEMPERATURE = 2.269


class IsingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Represent an Ising dataset stored as configurations and target values.

    Args:
        dataset_path: Path to a pickle file containing configurations and
            targets.
        target_dtype: PyTorch dtype used for returned targets.
    """

    def __init__(self, dataset_path: str | Path, target_dtype: torch.dtype) -> None:
        with Path(dataset_path).open("rb") as file:
            configurations, targets = pickle.load(file)
        self.configurations = np.asarray(configurations, dtype=np.float32)
        self.targets = np.asarray(targets)
        self.target_dtype = target_dtype

    def __len__(self) -> int:
        """Return the number of configurations in the dataset.

        Returns:
            Number of available configuration-target pairs.
        """
        return len(self.configurations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one channel-first configuration and its target.

        Args:
            index: Position of the requested sample.

        Returns:
            A tuple containing a spin tensor with one channel and its target
            tensor.
        """
        spins = torch.from_numpy(self.configurations[index]).unsqueeze(0)
        target = torch.tensor(self.targets[index], dtype=self.target_dtype)
        return spins, target


def generate_datasets(
    num_samples: int = 1_000,
    lattice_size: int = 20,
    monte_carlo_steps: int = 25_000,
    temperature_range: tuple[float, float] = (0.2, 4.0),
    classification_path: str | Path = "data/ising_configs_labels.pkl",
    regression_path: str | Path = "data/ising_configs_magnetization.pkl",
    seed: int | None = None,
) -> None:
    """Generate matching classification and regression datasets.

    Class 0 represents the high-temperature phase and class 1 represents the
    low-temperature phase. Regression targets are absolute magnetizations
    averaged over the latter half of each Markov chain.

    Args:
        num_samples: Number of temperatures and configurations to generate.
        lattice_size: Length of one side of the square spin lattice.
        monte_carlo_steps: Number of Metropolis updates per configuration.
        temperature_range: Inclusive lower and upper bounds for temperatures.
        classification_path: Output path for configuration-label pairs.
        regression_path: Output path for configuration-magnetization pairs.
        seed: Optional seed for reproducible dataset generation.
    """
    temperatures = np.linspace(*temperature_range, num_samples)
    initial_spins = np.ones((lattice_size, lattice_size), dtype=np.int8)
    rng = np.random.default_rng(seed)
    configurations: list[np.ndarray] = []
    phase_labels: list[int] = []
    magnetizations: list[float] = []

    for sample_index, temperature in enumerate(temperatures, start=1):
        _, magnetization_history, final_spins = metropolis(
            initial_spins.copy(),
            temperature,
            steps=monte_carlo_steps,
            seed=int(rng.integers(2**32)),
        )
        configurations.append(final_spins)
        magnetizations.append(float(np.abs(magnetization_history[monte_carlo_steps // 2 :]).mean()))
        phase_labels.append(int(temperature <= CRITICAL_TEMPERATURE))

        if sample_index % 100 == 0 or sample_index == num_samples:
            print(f"Generated {sample_index}/{num_samples} configurations.")

    save_pickle([configurations, phase_labels], classification_path)
    save_pickle([configurations, magnetizations], regression_path)


if __name__ == "__main__":
    generate_datasets()
