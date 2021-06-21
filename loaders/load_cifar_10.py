from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


def load_cifar_10():
    """"
    A helper function to load  the CIFAR-10 dataset and normalize it
    hot encode the label attributes.
    """

    # Number of classes.

    # Load the dataset.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Scale the pixel value to `-1 and 1` range.
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Change the label into one-hot-encoder vector
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
