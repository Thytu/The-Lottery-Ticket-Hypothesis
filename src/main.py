from modulefinder import Module
from LeNet import LeNet
from data_handler import get_data_load
from training import train_model, test_model
from torch.cuda import is_available as cuda_is_available
from torch import device as get_device, sum as torch_sum
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.nn.utils import prune
from copy import deepcopy
from ploting import plot_losses, plot_accuracies


NB_ITER = 8
NB_EPOCHS = 30
BATCH_SIZE = 64
DEFAULT_SPARSITY_RATE = 0.2
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

model = LeNet().to(DEVICE)
INITIAL_WEIGHTS = deepcopy(model.state_dict())

criterion = CrossEntropyLoss()
optimizer = Adam(
    params=model.parameters(),
    lr=3e-4,
)

losses = {}
accuracies = {}


def get_sparsity(model: Module) -> float:
    """
    Calculate the sparsity level given the training iteration step

    Args:
        model (Module): model the calculate the sparisty of (expects LeNet architecture)

    Returns:
        float: sparsity level given the training iteration step
    """

    return 100. * float(
        torch_sum(model.feature_extractor[0].weight == 0)
        + torch_sum(model.feature_extractor[3].weight == 0)
        + torch_sum(model.classifier[0].weight == 0)
        + torch_sum(model.classifier[2].weight == 0)
        + torch_sum(model.classifier[-1].weight == 0)
    ) / float(
        model.feature_extractor[0].weight.nelement()
        + model.feature_extractor[3].weight.nelement()
        + model.classifier[0].weight.nelement()
        + model.classifier[2].weight.nelement()
        + model.classifier[-1].weight.nelement()
    )


for training_iteration in tqdm(range(1, NB_ITER + 1), total=NB_ITER):
    pbar = tqdm(range(1, NB_EPOCHS + 1), total=NB_EPOCHS, leave=False)

    losses[get_sparsity(model=model)] = {}
    accuracies[get_sparsity(model=model)] = {}

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

        losses[get_sparsity(model=model)][iter_nb] = test_loss
        accuracies[get_sparsity(model=model)][iter_nb] = test_acc

        pbar.set_description(f"(iter: {iter_nb}) {train_loss=:.2f} {train_acc=:.2f} {test_loss=:.2f} {test_acc=:.2f}")

    parameters_to_prune = (
        (model.feature_extractor[0], 'weight'),
        (model.feature_extractor[3], 'weight'),
        (model.classifier[0], 'weight'),
        (model.classifier[2], 'weight'),
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=DEFAULT_SPARSITY_RATE if training_iteration < 3 else DEFAULT_SPARSITY_RATE * (training_iteration / 2),
    )

    # Reset weights
    reseted_weights = model.state_dict()
    for param in reseted_weights.keys():
        if param.split(".")[-1].replace("_orig", "") in ("bias", "weight"):
            reseted_weights[param] = deepcopy(INITIAL_WEIGHTS[param.replace("_orig", "")])

    model.load_state_dict(reseted_weights)


plot_losses(losses)
plot_accuracies(accuracies)
