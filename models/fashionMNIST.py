from torch import nn
import torch.nn.functional as F
from params import *


class FashionMNIST(nn.Module):
    """
    The FashionMnist neural network to classify fashionMNIST images.
    """

    def __init__(self):
        """
        The class constructor.
        """

        # Call the __init__ constructor of the parent class
        super(FashionMNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size= 5 )
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5 )

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)

        self.out = nn.Linear(in_features=60, out_features=10)

        # define forward propagation

        def forward(self, t):
            """
            Forward propagation
            :param self:
            :param t:
            :return:
            """

            # conv1
            t = self.conv1(t)
            t = F.relu(t)
            t = F.max_pool2d(t, kernel_size=2, stride=2)

            # conv2
            t = self.conv2(t)
            t = F.relu(t)
            t = F.max_pool2d(t, kernel_size=2, stride=2)

            # fc1
            t = t.reshape(-1, 12*4*4)
            t = self.fc1(t)
            t = F.relu(t)

            # fc2
            t = self.fc2(t)
            t = F.relu(t)

            # output
            t = self.out(t)

            return t



