from torch import nn

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
        self.flatten = nn.Flatten
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

        def forward(self, x):
            """
            Perform the forward pass
            :param self:
            :param x:
            :return:
            """

