from LeNet5 import LeNet5
from data_handler import get_data_load
from training import train_model, test_model
from torch.cuda import is_available as cuda_is_available
from torch import device as get_device
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.nn.utils import prune
from copy import deepcopy
from ploting import plot_losses, plot_accuracies


NB_ITER = 10
NB_EPOCHS = 160
BATCH_SIZE = 64
SPARSITY_RATE = 0.15
DEVICE = get_device("cuda" if cuda_is_available() else "cpu")


train_dataloader = get_data_load(
    split="train",
    batch_size=64,
    num_workers=4,
    subset=None
)

val_dataloader = get_data_load(
    split="val",
    batch_size=64,
    num_workers=4,
    subset=None
)

model = LeNet5(n_classes=10).to(DEVICE)
INITIAL_WEIGHTS = deepcopy(model.state_dict())

criterion = CrossEntropyLoss()
optimizer = Adam(
    params=model.parameters(),
    lr=3e-4,
)

losses = {}
accuracies = {}


def get_sparsity(training_iteration: int) -> float:
    """
    Calculate the sparsity level given the training iteration step

    Args:
        training_iteration (int): training iteration step

    Returns:
        float: sparsity level given the training iteration step
    """

    sparsity = 1.0

    for _ in range(training_iteration - 1):
        sparsity = sparsity * (1 - SPARSITY_RATE)

    return sparsity


for training_iteration in tqdm(range(1, NB_ITER + 1), total=NB_ITER):
    pbar = tqdm(range(1, NB_EPOCHS + 1), total=NB_EPOCHS, leave=False)

    losses[get_sparsity(training_iteration)] = {}
    accuracies[get_sparsity(training_iteration)] = {}

    for epoch in pbar:
        train_loss, train_acc = train_model(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE
        )

        test_loss, test_acc = test_model(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            device=DEVICE
        )

        iter_nb = epoch * len(train_dataloader)

        losses[get_sparsity(training_iteration)][iter_nb] = test_loss
        accuracies[get_sparsity(training_iteration)][iter_nb] = test_acc

        pbar.set_description(f"(iter: {iter_nb}) {train_loss=:.2f} {train_acc=:.2f} {test_loss=:.2f} {test_acc=:.2f}")

    # TODO: verify that pruning is cumulative
    parameters_to_prune = (
        (model.feature_extractor[0], 'weight'),
        (model.feature_extractor[3], 'weight'),
        (model.feature_extractor[6], 'weight'),
        (model.classifier[0], 'weight'),
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=SPARSITY_RATE,
    )

    # Reset weights
    reseted_weights = model.state_dict()
    for param in reseted_weights.keys():
        if param.split(".")[-1].replace("_orig", "") in ("bias", "weight"):
            reseted_weights[param] = INITIAL_WEIGHTS[param.replace("_orig", "")]

    model.load_state_dict(reseted_weights)


plot_losses(losses)
plot_accuracies(accuracies)
