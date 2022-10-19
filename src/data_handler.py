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


def get_data_loader(split: str, **kwargs) -> DataLoader:
    """
    Return a dataloader for the MNIST dataset

    Args:
        split (str): either train of val

    Returns:
        DataLoader: MNIST dataloader
    """

    dataset = MNIST(
        root=kwargs.get("root", './data'),
        train=(split == "train"),
        transform=get_transform(),
        download=True
    )

    if kwargs.get("subset", None):
        dataset = Subset(
            dataset=dataset,
            indices=torch_randperm(len(dataset))[:kwargs.get("subset")]
        )

    return DataLoader(
        dataset,
        batch_size=kwargs.get("batch_size", 32),
        shuffle=True,
        num_workers=kwargs.get("num_workers", 1)
    )
