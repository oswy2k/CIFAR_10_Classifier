# External Libraries Imports #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as datasets
import torchvision.transforms as T

# Main dataset Configuration #
DATA_PATH = "./Cifar10_Data"
NUM_TRAIN = 50000
NUM_VAL = 5000
NUM_TEST = 5000


def Cifar10_dataset_Generator(
    transforms: T.Compose,
    minibatch_size: int = 32,
    download: bool = True,
    log: bool = True,
) -> [DataLoader, DataLoader, DataLoader]:
    """
    Parameters
    ----------
    transforms : T.Compose
    minibatch_size : int = 32, optional
    download : bool = True, optional
    log : bool = True, optional
    Returns
    ----------
    [DataLoader, DataLoader, DataLoader]

    Notes
    ----------
    Downloads or unpacks Cifar10 dataset and makes the different dataloaders based on
    given transforms.
    """

    # Download Train dataset
    train_data_cifar10 = datasets.CIFAR10(
        DATA_PATH, train=True, download=download, transform=transforms
    )
    # Download Validation set
    validation_data_cifar10 = datasets.CIFAR10(
        DATA_PATH, train=False, download=download, transform=transforms
    )
    # Download Test set
    test_data_cifar10 = datasets.CIFAR10(
        DATA_PATH, train=False, download=download, transform=transforms
    )

    # Create Dataloader for Training
    train_dataLoader = DataLoader(
        train_data_cifar10,
        batch_size=minibatch_size,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
    )

    # Create Dataloader for Validation
    validation_dataLoader = DataLoader(
        validation_data_cifar10,
        batch_size=minibatch_size,
        sampler=sampler.SubsetRandomSampler(range(NUM_VAL)),
    )

    # Create Dataloader for Testing
    test_dataLoader = DataLoader(
        test_data_cifar10,
        batch_size=minibatch_size,
        sampler=sampler.SubsetRandomSampler(range(NUM_VAL, len(test_data_cifar10))),
    )

    if log:
        Verify_data(
            train_data_cifar10, "\n", validation_data_cifar10, "\n", test_data_cifar10
        )

    return [train_dataLoader, validation_dataLoader, test_dataLoader]


def Verify_data(*argv) -> None:
    """
    Parameters
    ----------
    argv : any

    Returns
    ----------
    None

    Notes
    ----------
    Takes all values given and prints them to the console.
    """

    for arguments in argv:
        print(arguments)
