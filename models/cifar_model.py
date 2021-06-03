from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import os
import numpy as np
import pickle
from utils import split_nbr


class CifarModel:
    """
    The Cifar helper class to train the model.
    """
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernel_size,
                 conv_strides,
                 z_shape,
                 use_batch_norm=True,
                 use_dropout=True):
        """
        The class constructor.
        """
        self.name = 'model'
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.n_conv_layer = len(conv_filters)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.z_shape = z_shape

        # build the architecture of the model
        self._build()

    def _build(self):
        """
        Build the model.
        :return: None
        """

        model_input = Input(shape=self.input_shape, name='model_input')
        x = model_input

        for i in range(self.n_conv_layer):
            x = Conv2D(filters=self.conv_filters[i],
                       kernel_size=self.conv_kernel_size[i],
                       strides=self.conv_strides[i], padding='same')(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(rate=0.5)(x)

        model_output = Dense(units=self.z_shape, activation='softmax', name='output')(x)

        # Define the model
        self.model = Model(model_input, model_output)

    def compile(self, lr):
        """
        Compile the model with an optimizer and a loss function.
        :param lr: double
            The learning rate.
        :return: None
        """

        self.learning_rate = lr

        # The optimized
        optimizer = Adam(learning_rate=lr)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer, metrics=['accuracy'])

    def train(self, x_train, y_train, location, epochs=10, batch_size=64, shuffle=True):
        """
        Train the model with training set.
        :return: None
        """
        # Store the learned weights.
        checkpoint = ModelCheckpoint(os.path.join('models_info', location, 'weights/weights.h5'),
                                     save_weights_only=True, verbose=1)
        self.model.fit(x_train, y_train, batch_size, epochs, shuffle, callbacks=checkpoint)

    def _plot_model(self, location):
        """
        Plot the model and save it.
        :param location: String
            The absolute path where to store the model summary
        :return: None
        """
        plot_model(self.model, to_file=os.path.join('models_info', location, 'viz/model.png'))

    def evaluate(self, x, y):
        """
        Evaluate the model on unseen data.
        :param x: array_like
            The test data
        :param y: array_like
            array one hot encoded vector.
        :return: None
        """

        self.model.evaluate(x, y)

    def predict(self, img):
        """
        Predict the class given an image.
        :param img: array_like
        :return: None
        """
        return self.model.predict(img, 1)

    def save(self, location):
        """
        Save the model alongside with its representation.
        :param location: String
            The absolute path where to store the model summary
        :return: None
        """

        # Check if the model directory exists
        if not os.path.exists(os.path.join('models_info', location)):
            os.makedirs(os.path.join('models_info', location))
            os.makedirs(os.path.join('models_info', location, 'viz'))
            os.makedirs(os.path.join('models_info', location, 'weights'))

        # Store the model's parameters
        with open(os.path.join('models_info', location, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_shape,
                self.conv_filters,
                self.conv_kernel_size,
                self.conv_strides,
                self.z_shape,
                self.use_batch_norm,
                self.use_dropout
            ], f)

        # Plot the model
        self._plot_model(location)

    def plot_predictions(self, x, y, nb_imgs=10, random_state=True):
        """
        plot the predicted label with its true-ground.
        :param x: array_like
            The test set.
        :param y: array_like
            The ground truth labels.
        :param nb_imgs: integer
            Number of images to plot
        :param random_state: boolean
            True for reproducible output.
        :return: None
        """
        # colors for predicted labels.
        colors = {
            'true': 'g',
            'false': 'r'
        }
        # labels to class name
        classes = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck'])

        # Make predictions
        y_preds = self.model.predict(x, nb_imgs)

        actual_classes = classes[np.argmax(y, axis=-1)]
        predicted_classes = classes[np.argmax(y_preds, axis=-1)]

        # Create a figure
        fig = plt.figure(figsize=(15, 5))

        # adjust the space between subplots
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        # Split the number of images provided
        n_rows, n_cols, _ = split_nbr(nb_imgs)
        for i in range(nb_imgs):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            act_cls = actual_classes[i]
            pred_cls = predicted_classes[i]

            # check whether the predicted label is equal to actual one.
            if act_cls == pred_cls:
                color = colors['true']
            else:
                color = colors['false']

            # label each sample
            ax.text(0, -0.25, 'actual = ' + str(act_cls), ha='left', transform=ax.transAxes)
            ax.text(0, -0.35, 'predicted = ' + str(pred_cls), ha='left', transform=ax.transAxes, color=color)
            ax.imshow(x[i])

    def load_weights(self, location):
        """
        Load the weights
        :param location: location where the weights parameters are stored.
        :return: None
        """

        # load saved weights
        self.model.load_weights(os.path.join('models_info', location))
