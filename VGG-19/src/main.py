from tqdm import tqdm
from VGG19 import VGG19
from copy import deepcopy
from torch.optim import SGD
from torch.nn.utils import prune
from data_handler import get_data_loaders
from training import train_model, test_model
from torch.nn import CrossEntropyLoss, Module
from ploting import plot_experiment
from torch.cuda import is_available as cuda_is_available
from torch import device as get_device, sum as torch_sum


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

        + torch_sum(model.features_extractor_block_1[0].weight == 0)
        + torch_sum(model.features_extractor_block_1[3].weight == 0)

        + torch_sum(model.features_extractor_block_2[0].weight == 0)
        + torch_sum(model.features_extractor_block_2[3].weight == 0)

        + torch_sum(model.features_extractor_block_3[0].weight == 0)
        + torch_sum(model.features_extractor_block_3[3].weight == 0)
        + torch_sum(model.features_extractor_block_3[6].weight == 0)
        + torch_sum(model.features_extractor_block_3[9].weight == 0)

        + torch_sum(model.features_extractor_block_4[0].weight == 0)
        + torch_sum(model.features_extractor_block_4[3].weight == 0)
        + torch_sum(model.features_extractor_block_4[6].weight == 0)
        + torch_sum(model.features_extractor_block_4[9].weight == 0)

        + torch_sum(model.features_extractor_block_5[0].weight == 0)
        + torch_sum(model.features_extractor_block_5[3].weight == 0)
        + torch_sum(model.features_extractor_block_5[6].weight == 0)
        + torch_sum(model.features_extractor_block_5[9].weight == 0)
    ) / float(
        model.classifier[0].weight.nelement()

        + model.features_extractor_block_1[0].weight.nelement()
        + model.features_extractor_block_1[3].weight.nelement()

        + model.features_extractor_block_2[0].weight.nelement()
        + model.features_extractor_block_2[3].weight.nelement()

        + model.features_extractor_block_3[0].weight.nelement()
        + model.features_extractor_block_3[3].weight.nelement()
        + model.features_extractor_block_3[6].weight.nelement()
        + model.features_extractor_block_3[9].weight.nelement()

        + model.features_extractor_block_4[0].weight.nelement()
        + model.features_extractor_block_4[3].weight.nelement()
        + model.features_extractor_block_4[6].weight.nelement()
        + model.features_extractor_block_4[9].weight.nelement()

        + model.features_extractor_block_5[0].weight.nelement()
        + model.features_extractor_block_5[3].weight.nelement()
        + model.features_extractor_block_5[6].weight.nelement()
        + model.features_extractor_block_5[9].weight.nelement()
    )


def main(nb_pruning_iter, max_epoch, p_conv):

    DEVICE = get_device("cuda" if cuda_is_available() else "cpu")

    model = VGG19().to(DEVICE)
    INITIAL_WEIGHTS = deepcopy(model.state_dict())

    criterion = CrossEntropyLoss()
    optimizer = SGD(
        params=model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001
    )

    # TODO: use val loader for evaluation early stop
    (train_dataloader, test_dataloader, _) = get_data_loaders(batch_size=64, num_workers=4)

    test_losses = {}
    test_accuracies = {}

    # 'n' is the paper represents the number of pruning iterations
    for n in tqdm(range(1, nb_pruning_iter + 1), total=nb_pruning_iter, leave=False):
        pbar = tqdm(range(max_epoch), leave=False)

        current_sparisty = get_sparsity(model=model)

        test_losses[current_sparisty] = {}
        test_accuracies[current_sparisty] = {}

        for epoch in pbar:

            # learning rate scheduling
            if epoch + 1 in [80, 120]:
                for g in optimizer.param_groups:
                    g['lr'] /= 10

            train_loss, train_acc = train_model(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=DEVICE
            )

            test_loss, test_acc = test_model(
                model=model,
                dataloader=test_dataloader,
                criterion=criterion,
                device=DEVICE
            )

            test_losses[current_sparisty][epoch] = test_loss
            test_accuracies[current_sparisty][epoch] = test_acc

            pbar.set_description(f"{train_loss=:.2f} {train_acc=:.2f} {test_loss=:.2f} {test_acc=:.2f}")
            pbar.update(1)

        pbar.close()

        pruning_rate_conv = p_conv ** (1 / n)

        parameters_to_prune = (
            (model.features_extractor_block_1[0], "weight"),
            (model.features_extractor_block_1[3], "weight"),

            (model.features_extractor_block_2[0], "weight"),
            (model.features_extractor_block_2[3], "weight"),

            (model.features_extractor_block_3[0], "weight"),
            (model.features_extractor_block_3[3], "weight"),
            (model.features_extractor_block_3[6], "weight"),
            (model.features_extractor_block_3[9], "weight"),

            (model.features_extractor_block_4[0], "weight"),
            (model.features_extractor_block_4[3], "weight"),
            (model.features_extractor_block_4[6], "weight"),
            (model.features_extractor_block_4[9], "weight"),

            (model.features_extractor_block_5[0], "weight"),
            (model.features_extractor_block_5[3], "weight"),
            (model.features_extractor_block_5[6], "weight"),
            (model.features_extractor_block_5[9], "weight"),
        )

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_rate_conv,
        )

        # Reset weights
        reseted_weights = deepcopy(model.state_dict())
        for param in reseted_weights.keys():
            if param.split(".")[-1].replace("_orig", "") in ("bias", "weight"):
                reseted_weights[param] = deepcopy(INITIAL_WEIGHTS[param.replace("_orig", "")])

        model.load_state_dict(reseted_weights)

    return test_losses, test_accuracies


if __name__ == "__main__":
    P_CONV = 0.2

    NB_RUN = 3
    NB_PRUNING_ITER = 7
    EPOCHS = 160

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
            max_epoch=EPOCHS,
            p_conv=P_CONV,
        )

        test_losses.append(test_losses_in_run)
        test_accuracies.append(test_accuracies_in_run)

    plot_experiment(test_losses, test_accuracies)
