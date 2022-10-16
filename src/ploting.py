from typing import Dict, List
from matplotlib import pyplot as plt


def plot_losses(losses: Dict[float, Dict[float, List[float]]]) -> None:
    """
    Plot the evolution of the loss regarding the sparsity level and iteration step

    Args:
        losses (Dict[float, Dict[float, List[float]]]): Dict containing the losses regarding the sparsity level and iteration step
    """

    plt.clf()

    sparsity_levels = [round(sparsity_level, 2) for sparsity_level in losses.keys()]

    for sparsity_level, key in zip(sparsity_levels, losses.keys()):
        plt.plot(list(losses[key].keys()), list(losses[key].values()), '+--', label=f"Sparsity: {sparsity_level}")

    plt.xlabel("Training iterations")
    plt.ylabel("Loss on the test set")

    plt.legend(loc='best')
    plt.savefig("losses.png")


def plot_accuracies(accuracies: Dict[float, Dict[float, List[float]]]) -> None:
    """
    Plot the evolution of the accuracy regarding the sparsity level and iteration step

    Args:
        accuracies (Dict[float, Dict[float, List[float]]]): Dict containing the accuracies regarding the sparsity level and iteration step
    """

    plt.clf()

    sparsity_levels = [round(sparsity_level, 2) for sparsity_level in accuracies.keys()]

    for sparsity_level, key in zip(sparsity_levels, accuracies.keys()):
        plt.plot(list(accuracies[key].keys()), list(accuracies[key].values()), '+--', label=f"Sparsity: {sparsity_level}")

    plt.xlabel("Training iterations")
    plt.ylabel("Accuracy on the test set")

    plt.legend(loc='best')
    plt.savefig("accuracies.png")