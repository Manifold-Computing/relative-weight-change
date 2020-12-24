from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from utils.transforms import train_transforms, val_transforms


def train_dataset(n_workers, name='CIFAR10', b_size=32):
    """ Function returns the train dataset used for training

    Args:
        n_workers ([int]): Number of workers used to load the dataset
        name (str, optional): Name of the dataset to load. Defaults to 'CIFAR10'.
        b_size (int, optional): Batch Size for the dataset. Defaults to 32.

    Returns:
        torch.DataLoader: PyTorch dataloader used in training loop
    """

    if name == "CIFAR10":
        dataset = CIFAR10("../", train=True, download=True, transform=train_transforms())
    elif name == "CIFAR100":
        dataset = CIFAR100("../", train=True, download=True, transform=train_transforms())
    elif name == "MNIST":
        dataset = MNIST("../", train=True, download=True, transform=train_trasforms())
    elif name == "FMNIST":
        dataset = FashionMNIST("../", train=True, download=True, transform=train_trasforms())

    return DataLoader(dataset, batch_size=b_size, num_workers=n_workers)


def val_dataset(n_workers,name='CIFAR10', b_size=32):
    """ Function returns the dataset used for validaton

    Args:
        n_workers ([int]): Number of workers used to load the dataset
        name (str, optional): Name of the dataset to load. Defaults to 'CIFAR10'.
        b_size (int, optional): Batch size for the dataset. Defaults to 32.

    Returns:
        torch.DataLoader: PyTorch dataloader used in validation loop
    """

    if name == "CIFAR10":
        dataset = CIFAR10("../", train=False, download=True, transform=val_transforms())
    elif name == "CIFAR100":
        dataset = CIFAR100("../", train=False, download=True, transform=val_transforms())
    elif name == "MNIST":
        dataset = MNIST("../", train=False, download=True, transform=val_transforms())
    elif name == "FMNIST":
        dataset = FashionMNIST("../", train=False, download=True, transform=val_transforms())

    return DataLoader(dataset, batch_size=b_size, num_workers=n_workers)
