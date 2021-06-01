import pickle
import os


def load_model(model_class, location):
    """
    Load the pre-trained model from disk.
    :param model_class: class
        Model class used to construct the model.
    :param location: string
        The location where params and weights learned by the model live.
    :return: instance of `model_class`
    """

    # load model params
    with open(os.path.join('models_info', location, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)

    # destructuring the loaded parameters
    model = model_class(*params)

    # load weights
    model.load_weights(os.path.join('models_info', location, 'weights/weights.h5'))

    return model
