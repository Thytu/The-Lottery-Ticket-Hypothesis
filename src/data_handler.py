from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torch import randperm as torch_randperm


def get_transform(split: str) -> transforms.Compose:
    """
    Get Compose transform for the MNIST dataset and LeNet5 network

    Args:
        split (str): either train or val

    Raises:
        RuntimeError: Unknown split value

    Returns:
        transforms.Compose: transform for the MNIST dataset and LeNet5 network
    """

    normalize_transform = transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    )

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
            normalize_transform,
        ])

    elif split == "val":
        return transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize_transform,
        ])

    raise RuntimeError(f"Unknown split value {split}, must be either 'train' or 'val'")


def get_data_load(split: str, **kwargs) -> DataLoader:
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
        transform=get_transform(split),
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
