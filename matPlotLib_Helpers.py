# External Libraries Imports #
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

# System Imports #
import random


# Matplotlib plot Tensor as Image #
def Plot_figure(dataLoader: DataLoader) -> None:
    """
    Parameters
    ----------
    dataLoader : DataLoader

    Returns
    ----------
    None

    Notes
    ----------
    Plot a random image from the dataloader given.
    """

    # Get loader reference for classes #
    data_Classes = dataLoader.dataset.classes

    # Get Random image and Label #
    random_Index = np.random.randint(len(dataLoader))

    label = data_Classes[dataLoader.dataset[random_Index][1]]
    image = dataLoader.dataset[random_Index][0]

    print(f"The image Class is: {label}")

    # Normalize image colors for matplotlib #
    image = (image - image.min()) / (image.max() - image.min())

    # Plot image #
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.axis("off")
    plt.show()


# MatplotLib plot Tensor as Grid #
def Plot_figures_grid(
    dataLoader: DataLoader, samples: int = 10, gridsize: [int, int] = [10, 10]
) -> None:
    """
    Parameters
    ----------
    dataLoader : DataLoader
    samples : int = 10, optional
    gridsize : [int, int] = [10, 10], optional

    Returns
    ----------
    None

    Notes
    ----------
    Plot a grid of random images from the dataloader given.
    """

    # Get loader reference for its classes #
    data_Classes = dataLoader.dataset.classes

    plt.figure(figsize=gridsize)

    # Iterate the classes looking for randomized image to show #
    for label, sample in enumerate(data_Classes):
        # Randomize indices for images
        class_indices = np.flatnonzero(label == np.array(dataLoader.dataset.targets))
        sample_indices = np.random.choice(class_indices, samples, replace=False)

        for i, index in enumerate(sample_indices):
            plot_index = i * len(data_Classes) + label + 1
            plt.subplot(samples, len(data_Classes), plot_index)
            plt.imshow(dataLoader.dataset.data[index])
            plt.axis("off")

            if i == 0:
                plt.title(sample)

    # Plot image #
    plt.show()


def Plot_minibatch_loss(logging_Dict: dict, loss_per_batch_label: str) -> None:
    """
    Parameters
    ----------
    logging_Dict : dict
    loss_per_batch_label : str

    Returns
    ----------
    None

    Notes
    ----------
    Plot the Loss and average loss of the Trained Network, given a logging dictionary and its key.
    """

    loss_minibatch_list = logging_Dict[loss_per_batch_label]

    plt.plot(loss_minibatch_list, label="Minibatch loss Function")

    # Calculate Average #
    plt.plot(
        np.convolve(
            loss_minibatch_list,
            np.ones(
                200,
            )
            / 200,
            mode="valid",
        ),
        label="Running average",
    )

    # Plot Graph #
    plt.ylabel("Cross Entropy")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()


def Plot_accuracy_epochs(
    logging_Dict: dict, total_epochs: int, accuracy_per_epoch_labels: [str, str]
) -> None:
    """
    Parameters
    ----------
    logging_Dict : dict
    total_epochs : int
    accuracy_per_epoch_labels : [str, str]

    Returns
    ----------
    None

    Notes
    ----------
    Plot the accuracy of the Trained Network for the train and validation datasets, given a logging dictionary and their keys.
    """

    # Arrange data #
    plt.plot(
        np.arange(1, total_epochs + 1),
        logging_Dict[accuracy_per_epoch_labels[0]],
        label="Training",
    )
    plt.plot(
        np.arange(1, total_epochs + 1),
        logging_Dict[accuracy_per_epoch_labels[1]],
        label="Validation",
    )

    # Plot Graph #
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def Plot_confusion_matrix(matrix: any, labels: [], text_show: str) -> None:
    """
    Parameters
    ----------
    matrix : any
    labels : []
    text_show : str

    Returns
    ----------
    None

    Notes
    ----------
    Plots the confusion matrix of the trained network.
    """

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            text = ax.text(j, i, matrix[i, j], ha="center", va="center", color="w")

    ax.set_title(text_show)
    fig.tight_layout()
    plt.show()
