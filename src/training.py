from tqdm import tqdm
from torch.nn import Module
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from torch import device as torch_device, eq as torch_eq, argmax as torch_argmax, no_grad as torch_no_grad
from torch.cuda import is_available as cuda_is_available
from torch.optim import Optimizer


def calc_accurary(preds, y_true) -> float:
    return (sum(torch_eq(torch_argmax(preds, dim=1), y_true)) / len(preds)).item()


def train_model(
    model: Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: Module,
    training_iteration: int,
    max_training_iter: int,
    device: Optional[torch_device] = None,
) -> Tuple[float, float]:
    """
    Train the model given the provided data, optimizer and criterion

    Args:
        model (Module): model to train
        dataloader (DataLoader): data to train the model on
        optimizer (Optimizer): optimizer to use
        criterion (Module): criterion to optimize
        training_iteration (int): current training iteration
        max_training_iter (int): used to stop the training when >= training_iteration max_training_iter
        device (Optional[torch_device], optional): device to use. Defaults to None.

    Returns:
        Tuple[float, float]: loss and accuracy at the end of the training
    """

    model.train()

    if not device:
        device = torch_device("cuda" if cuda_is_available() else "cpu")

    step_losses = []
    step_accuracies = []

    pbar = tqdm(dataloader, total=len(dataloader), leave=False)

    for data in pbar:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        training_iteration += 1

        step_losses.append(loss.item())
        step_accuracies.append(calc_accurary(outputs, labels))

        pbar.set_description(f'[train] loss: {sum(step_losses) / len(step_losses):.3f} acc: {sum(step_accuracies) / len(step_accuracies):.3f}')

        if training_iteration >= max_training_iter:
            break

    return sum(step_losses) / len(step_losses), sum(step_accuracies) / len(step_accuracies), training_iteration


def test_model(
    model: Module,
    dataloader: DataLoader,
    criterion: Module,
    device: Optional[torch_device] = None
) -> Tuple[float, float]:
    """
    test the model given the provided data and criterion

    Args:
        model (Module): model to test
        dataloader (DataLoader): data to test the model on
        criterion (Module): criterion to optimize
        device (Optional[torch_device], optional): device to use. Defaults to None.

    Returns:
        Tuple[float, float]: loss and accuracy at the end of the testing
    """

    model.eval()

    if not device:
        device = torch_device("cuda" if cuda_is_available() else "cpu")

    step_losses = []
    step_accuracies = []

    pbar = tqdm(dataloader, total=len(dataloader), leave=False)

    with torch_no_grad():
        for data in pbar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            step_losses.append(loss.item())
            step_accuracies.append(calc_accurary(outputs, labels))

            pbar.set_description(f'[eval] loss: {sum(step_losses) / len(step_losses):.3f} acc: {sum(step_accuracies) / len(step_accuracies):.3f}')

    return sum(step_losses) / len(step_losses), sum(step_accuracies) / len(step_accuracies)
