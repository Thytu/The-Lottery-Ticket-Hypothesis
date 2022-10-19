from tqdm import tqdm
from LeNet import LeNet
from copy import deepcopy
from torch.optim import Adam
from torch.nn.utils import prune
from data_handler import get_data_loaders
from training import train_model, test_model
from torch.nn import CrossEntropyLoss, Module
from ploting import plot_losses, plot_accuracies
from torch.cuda import is_available as cuda_is_available
from torch import device as get_device, sum as torch_sum


P = 0.2
BATCH_SIZE = 60
NB_PRUNING_ITER = 9
MAX_TRAINING_ITER = 15_000
DEVICE = get_device("cuda" if cuda_is_available() else "cpu")


model = LeNet().to(DEVICE)
INITIAL_WEIGHTS = deepcopy(model.state_dict())

criterion = CrossEntropyLoss()
optimizer = Adam(
    params=model.parameters(),
    lr=1.2e-3,
)

# TODO: use val loader for evaluation early stop
(train_dataloader, test_dataloader, _) = get_data_loaders(batch_size=64, num_workers=4)

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
    ) / float(
        model.classifier[0].weight.nelement()
        + model.classifier[2].weight.nelement()
        + model.classifier[-1].weight.nelement()
    )


# 'n' is the paper represents the number of pruning iterations
for n in tqdm(range(1, NB_PRUNING_ITER + 1), total=NB_PRUNING_ITER):
    pbar = tqdm(total=MAX_TRAINING_ITER, leave=False)

    test_losses[get_sparsity(model=model)] = {}
    test_accuracies[get_sparsity(model=model)] = {}

    training_iteration = 0

    while training_iteration < MAX_TRAINING_ITER:

        last_training_iteration = training_iteration
        train_loss, train_acc, training_iteration = train_model(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            training_iteration=training_iteration,
            max_training_iter=MAX_TRAINING_ITER,
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

        pbar.update(training_iteration - last_training_iteration)
        pbar.set_description(f"{train_loss=:.2f} {train_acc=:.2f} {test_loss=:.2f} {test_acc=:.2f}")

    pruning_rate = P ** (1 / n)

    prune.l1_unstructured(model.classifier[0], name="weight", amount=pruning_rate)
    prune.l1_unstructured(model.classifier[2], name="weight", amount=pruning_rate)
    prune.l1_unstructured(model.classifier[-1], name="weight", amount=pruning_rate / 2)

    # Reset weights
    reseted_weights = deepcopy(model.state_dict())
    for param in reseted_weights.keys():
        if param.split(".")[-1].replace("_orig", "") in ("bias", "weight"):
            reseted_weights[param] = deepcopy(INITIAL_WEIGHTS[param.replace("_orig", "")])

    model.load_state_dict(reseted_weights)


plot_losses(test_losses)
plot_accuracies(test_accuracies)
