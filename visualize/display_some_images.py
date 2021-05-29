import numpy as np
from matplotlib import pyplot as plt
from utils import split_nbr


def display_some_images(x, y, num_imgs):
    """
    Display some images
    :param x: array_like
        The image array
    :param y: array_like
        array of labels
    :param num_imgs: integer
        Number of images to display.
    :return: None
    """

    # labels to class name
    classes = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    # The number of rows and columns in the figure
    n_rows, n_cols, _ = split_nbr(num_imgs)

    # Create the figure
    fig = plt.figure(figsize=(n_cols + 8, n_rows + 5))

    for i in range(num_imgs):
        fig.add_subplot(n_rows, n_cols, i + 1)
        plt.imshow(x[i], cmap=plt.cm.binary, interpolation='nearest')
        plt.title(classes[int(np.argmax(y[i]))])

