import torch.nn as nn
import torch.nn.functional as F


# Implementation of Custom net for Cifar 10 dataset #
class CNN_custom(nn.Module):
    """
    Parameters
    ----------
    number_of_classes = 10, optional

    Returns
    ----------
    CNN_custom class Object.

    Notes
    ----------
    Returns an Neural nerwork object.
    """

    def __init__(self, number_of_classes: int = 10):
        super().__init__()
        # Convolutional Layers Features #
        self.convolutional_features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=3),   # Input Layer, has three channels to convert rgb to different kernels, its size its bigger to have more features in it,
                                                                    # also its output is upscaled by the padding to mantain most of the features#
            nn.ReLU(inplace=True),                                  # Activation function #   
            nn.MaxPool2d(kernel_size=3, stride=1),                  # Max pooling layer, to reduce the size of the image, and gather characteristics for next convolutional layer#

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=3),  # Intermediate layer 1, reduced kerneel size for more precise features #
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=3),  # Intermediate layer 2, mantains kernel size and upscale number of kernels for average pooling fusing #
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Average Pooling Features #
        self.average_pooling = nn.AdaptiveAvgPool2d((8, 8))         # Fuse all characteristics in 8x8 image for linearizing and classifiying #

        # Linear Layers Classification Features #
        self.classifiying_features = nn.Sequential(                 # Linear layers to classify the image #
            nn.Dropout(0.1),
            nn.Linear(8 * 8 * 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, number_of_classes),
        )

    def forward(self, x):
        convolutional_output = self.convolutional_features(x)
        average_pooling_output = self.average_pooling(convolutional_output)
        average_output = average_pooling_output.view(
            average_pooling_output.size(0), 64 * 8 * 8
        )
        classificaiton_output = self.classifiying_features(average_output)
        return classificaiton_output
