from typing import Tuple
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torch import randperm as torch_randperm


def get_transform() -> transforms.Compose:
    """
    Get Compose transform for the MNIST dataset and LeNet5 network

    Returns:
        transforms.Compose: transform for the MNIST dataset and LeNet5 network
    """

    normalize_transform = transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    )

    return transforms.Compose([
        transforms.ToTensor(),
        normalize_transform,
    ])


def get_data_loaders(**kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set = MNIST(
        root=kwargs.get("root", './data'),
        train=True,
        transform=get_transform(),
        download=True
    )

    shuffled_indices = torch_randperm(len(train_set))

    val_set = Subset(
        dataset=train_set,
        indices=shuffled_indices[:5_000]
    )

    train_set = Subset(
        dataset=train_set,
        indices=shuffled_indices[5_000:]
    )

    test_set = MNIST(
        root=kwargs.get("root", './data'),
        train=False,
        transform=get_transform(),
        download=True
    )

    return DataLoader(
        dataset=train_set,
        batch_size=kwargs.get("batch_size", 32),
        shuffle=True,
        num_workers=kwargs.get("num_workers", 1)
    ), DataLoader(
        dataset=test_set,
        batch_size=kwargs.get("batch_size", 32),
        shuffle=True,
        num_workers=kwargs.get("num_workers", 1)
    ), DataLoader(
        dataset=val_set,
        batch_size=kwargs.get("batch_size", 32),
        shuffle=True,
        num_workers=kwargs.get("num_workers", 1)
    )