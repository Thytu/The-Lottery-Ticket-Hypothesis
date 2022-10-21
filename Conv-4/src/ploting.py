from typing import Dict, List
from matplotlib import pyplot as plt


def __plot_test_losses(
    losses: Dict[float, Dict[float, float]],
    mins: Dict[float, Dict[float, float]],
    maxs: Dict[float, Dict[float, float]],
) -> None:
    """
    Plot the evolution of the loss regarding the sparsity level and iteration step

    Args:
        losses (Dict[float, Dict[float, float]]): Dict containing the mean losses regarding the sparsity level and iteration step
        mins (Dict[float, Dict[float, float]]): Dict containing the min losses regarding the sparsity level and iteration step
        maxs (Dict[float, Dict[float, float]]): Dict containing the max losses regarding the sparsity level and iteration step
    """

    plt.clf()

    plt.figure(figsize=(20, 10))
    plt.tight_layout()

    sparsity_levels = [round(sparsity_level, 2) for sparsity_level in losses.keys()]

    for sparsity_level, key in zip(sparsity_levels, losses.keys()):
        yerr = (list(mins[key].values()), list(maxs[key].values()))

        plt.errorbar(list(losses[key].keys()), list(losses[key].values()), yerr=yerr, fmt='+--', label=f"{100 - sparsity_level:.2f}%")

    plt.xlabel("Training iterations")
    plt.ylabel("Loss on the test set")
    plt.title("Model's loss regarding the fraction of weights remaining in the network after pruning.")

    plt.legend(loc='best')
    plt.savefig("images/losses.png", bbox_inches='tight', pad_inches=0.1)


def __plot_test_accuracies(
    accuracies: Dict[float, Dict[float, float]],
    mins: Dict[float, Dict[float, float]],
    maxs: Dict[float, Dict[float, float]],
) -> None:
    """
    Plot the evolution of the accuracy regarding the sparsity level and iteration step

    Args:
        accuracies (Dict[float, Dict[float, float]]): Dict containing the mean accuracies regarding the sparsity level and iteration step
        mins (Dict[float, Dict[float, float]]): Dict containing the min accuracies regarding the sparsity level and iteration step
        maxs (Dict[float, Dict[float, float]]): Dict containing the max accuracies regarding the sparsity level and iteration step
    """

    plt.clf()

    plt.figure(figsize=(20, 10))
    plt.tight_layout()

    sparsity_levels = [round(sparsity_level, 2) for sparsity_level in accuracies.keys()]

    for sparsity_level, key in zip(sparsity_levels, accuracies.keys()):
        yerr = (list(mins[key].values()), list(maxs[key].values()))

        plt.errorbar(list(accuracies[key].keys()), list(accuracies[key].values()), yerr=yerr, fmt='+--', label=f"{100 - sparsity_level:.2f}%")

    plt.xlabel("Training iterations")
    plt.ylabel("Accuracy on the test set")
    plt.title("Model's accuracy regarding the fraction of weights remaining in the network after pruning.")


    plt.legend(loc='best')
    plt.savefig("images/accuracies.png", bbox_inches='tight', pad_inches=0.1)


def plot_experiment(
    test_losses: List[Dict[float, Dict[float, float]]],
    test_accuracies: List[Dict[float, Dict[float, float]]],
):
    """
    Plot the experiment's results (test loss, test accuracy, early stop)

    Args:
        test_losses (List[Dict[float, Dict[float, float]]]): evolution of the test loss in each run
        test_accuracies (List[Dict[float, Dict[float, float]]]): evolution of the test accuracy in each run
    """

    losses = {}
    min_losses = {}
    max_losses = {}

    accuracies = {}
    min_accuracies = {}
    max_accuracies = {}

    nb_run = len(test_losses)

    for sparsity in test_losses[0].keys():

        losses[sparsity] = {}
        min_losses[sparsity] = {}
        max_losses[sparsity] = {}

        accuracies[sparsity] = {}
        min_accuracies[sparsity] = {}
        max_accuracies[sparsity] = {}

        for training_iteration in test_losses[0][sparsity].keys():
            loss_run_values = [test_losses[run][sparsity][training_iteration] for run in range(nb_run)]
            acc_run_values = [test_accuracies[run][sparsity][training_iteration] for run in range(nb_run)]

            losses[sparsity][training_iteration] = sum(loss_run_values) / len(loss_run_values)
            min_losses[sparsity][training_iteration] = abs(losses[sparsity][training_iteration] - min(loss_run_values))
            max_losses[sparsity][training_iteration] = abs(losses[sparsity][training_iteration] - max(loss_run_values))

            accuracies[sparsity][training_iteration] = sum(acc_run_values) / len(acc_run_values)
            min_accuracies[sparsity][training_iteration] = abs(accuracies[sparsity][training_iteration] - min(loss_run_values))
            max_accuracies[sparsity][training_iteration] = abs(accuracies[sparsity][training_iteration] - max(loss_run_values))

    __plot_test_losses(losses, min_losses, max_losses)
    __plot_test_accuracies(accuracies, min_losses, max_losses)
