from typing import Dict, List
from matplotlib import pyplot as plt


def plot_losses(losses: Dict[float, Dict[float, List[float]]]) -> None:
    """
    Plot the evolution of the loss regarding the sparsity level and iteration step

    Args:
        losses (Dict[float, Dict[float, List[float]]]): Dict containing the losses regarding the sparsity level and iteration step
    """

    plt.clf()

    plt.figure(figsize=(20, 10))
    plt.tight_layout()

    sparsity_levels = [round(sparsity_level, 2) for sparsity_level in losses.keys()]

    for sparsity_level, key in zip(sparsity_levels, losses.keys()):
        plt.plot(list(losses[key].keys()), list(losses[key].values()), '+--', label=f"{int(100 - sparsity_level)}%")

    plt.xlabel("Training iterations")
    plt.ylabel("Loss on the test set")
    plt.title("Model's loss regarding the fraction of weights remaining in the network after pruning.")

    plt.legend(loc='best')
    plt.savefig("images/losses.png", bbox_inches='tight', pad_inches=0.1)


def plot_accuracies(accuracies: Dict[float, Dict[float, List[float]]]) -> None:
    """
    Plot the evolution of the accuracy regarding the sparsity level and iteration step

    Args:
        accuracies (Dict[float, Dict[float, List[float]]]): Dict containing the accuracies regarding the sparsity level and iteration step
    """

    plt.clf()

    plt.figure(figsize=(20, 10))
    plt.tight_layout()

    sparsity_levels = [round(sparsity_level, 2) for sparsity_level in accuracies.keys()]

    for sparsity_level, key in zip(sparsity_levels, accuracies.keys()):
        plt.plot(list(accuracies[key].keys()), list(accuracies[key].values()), '+--', label=f"{int(100 - sparsity_level)}%")

    plt.xlabel("Training iterations")
    plt.ylabel("Accuracy on the test set")
    plt.title("Model's accuracy regarding the fraction of weights remaining in the network after pruning.")


    plt.legend(loc='best')
    plt.savefig("images/accuracies.png", bbox_inches='tight', pad_inches=0.1)
