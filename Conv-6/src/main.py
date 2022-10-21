from tqdm import tqdm
from Conv6 import Conv6
from copy import deepcopy
from torch.optim import Adam
from torch.nn.utils import prune
from data_handler import get_data_loaders
from training import train_model, test_model
from torch.nn import CrossEntropyLoss, Module
from ploting import plot_experiment
from torch.cuda import is_available as cuda_is_available
from torch import device as get_device, sum as torch_sum


def main(nb_pruning_iter, max_training_iter, p_linear, p_conv):

    DEVICE = get_device("cuda" if cuda_is_available() else "cpu")

    model = Conv6().to(DEVICE)
    INITIAL_WEIGHTS = deepcopy(model.state_dict())

    criterion = CrossEntropyLoss()
    optimizer = Adam(
        params=model.parameters(),
        lr=3e-4,
    )

    # TODO: use val loader for evaluation early stop
    (train_dataloader, test_dataloader, _) = get_data_loaders(batch_size=60, num_workers=4)

    test_losses = {}
    test_accuracies = {}


    def get_sparsity(model: Module) -> float:
        """
        Calculate the sparsity level given the training iteration step

        Args:
            model (Module): model the calculate the sparisty of (expects LeNet architecture)

        Returns:
            float: sparsity level given the training iteration step
        """

        return 100. * float(
            torch_sum(model.classifier[0].weight == 0)
            + torch_sum(model.classifier[2].weight == 0)
            + torch_sum(model.classifier[-1].weight == 0)

            + torch_sum(model.features_extractor_block_1[0].weight == 0)
            + torch_sum(model.features_extractor_block_1[2].weight == 0)

            + torch_sum(model.features_extractor_block_2[0].weight == 0)
            + torch_sum(model.features_extractor_block_2[2].weight == 0)

            + torch_sum(model.features_extractor_block_3[0].weight == 0)
            + torch_sum(model.features_extractor_block_3[2].weight == 0)
        ) / float(
            model.classifier[0].weight.nelement()
            + model.classifier[2].weight.nelement()
            + model.classifier[-1].weight.nelement()

            + model.features_extractor_block_1[0].weight.nelement()
            + model.features_extractor_block_1[2].weight.nelement()

            + model.features_extractor_block_2[0].weight.nelement()
            + model.features_extractor_block_2[2].weight.nelement()

            + model.features_extractor_block_3[0].weight.nelement()
            + model.features_extractor_block_3[2].weight.nelement()
        )


    # 'n' is the paper represents the number of pruning iterations
    for n in tqdm(range(1, nb_pruning_iter + 1), total=nb_pruning_iter, leave=False):
        pbar = tqdm(total=max_training_iter, leave=False)

        test_losses[get_sparsity(model=model)] = {}
        test_accuracies[get_sparsity(model=model)] = {}

        training_iteration = 0

        while training_iteration < max_training_iter:

            last_training_iteration = training_iteration
            train_loss, train_acc, training_iteration = train_model(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                training_iteration=training_iteration,
                max_training_iter=max_training_iter,
                device=DEVICE
            )

            test_loss, test_acc = test_model(
                model=model,
                dataloader=test_dataloader,
                criterion=criterion,
                device=DEVICE
            )

            test_losses[get_sparsity(model=model)][training_iteration] = test_loss
            test_accuracies[get_sparsity(model=model)][training_iteration] = test_acc

            pbar.set_description(f"{train_loss=:.2f} {train_acc=:.2f} {test_loss=:.2f} {test_acc=:.2f}")
            pbar.update(training_iteration - last_training_iteration)

        pbar.close()

        pruning_rate_conv = p_conv ** (1 / n)
        prune.l1_unstructured(model.features_extractor_block_1[0], name="weight", amount=pruning_rate_conv)
        prune.l1_unstructured(model.features_extractor_block_1[2], name="weight", amount=pruning_rate_conv)

        prune.l1_unstructured(model.features_extractor_block_2[0], name="weight", amount=pruning_rate_conv)
        prune.l1_unstructured(model.features_extractor_block_2[2], name="weight", amount=pruning_rate_conv)

        prune.l1_unstructured(model.features_extractor_block_3[0], name="weight", amount=pruning_rate_conv)
        prune.l1_unstructured(model.features_extractor_block_3[2], name="weight", amount=pruning_rate_conv)

        pruning_rate_linear = p_linear ** (1 / n)
        prune.l1_unstructured(model.classifier[0], name="weight", amount=pruning_rate_linear)
        prune.l1_unstructured(model.classifier[2], name="weight", amount=pruning_rate_linear)
        prune.l1_unstructured(model.classifier[-1], name="weight", amount=pruning_rate_linear / 2)

        # Reset weights
        reseted_weights = deepcopy(model.state_dict())
        for param in reseted_weights.keys():
            if param.split(".")[-1].replace("_orig", "") in ("bias", "weight"):
                reseted_weights[param] = deepcopy(INITIAL_WEIGHTS[param.replace("_orig", "")])

        model.load_state_dict(reseted_weights)

    return test_losses, test_accuracies


if __name__ == "__main__":
    P_CONV = 0.15
    P_LINEAR = 0.2

    NB_RUN = 3
    NB_PRUNING_ITER = 7
    MAX_TRAINING_ITER = 25_000

    test_losses = []
    test_accuracies = []

    losses = {}
    min_losses = {}
    max_losses = {}

    accuracies = {}
    min_accuracies = {}
    max_accuracies = {}

    for run in tqdm(range(NB_RUN)):

        test_losses_in_run, test_accuracies_in_run = main(
            nb_pruning_iter=NB_PRUNING_ITER,
            max_training_iter=MAX_TRAINING_ITER,
            p_conv=P_CONV,
            p_linear=P_LINEAR,
        )

        test_losses.append(test_losses_in_run)
        test_accuracies.append(test_accuracies_in_run)

    plot_experiment(test_losses, test_accuracies)
