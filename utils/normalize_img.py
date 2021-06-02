def normalize_img(img):
    """
    Normalize the given images.
    :param img: array_like
    :return: array_like
        Normalized image.
    """

    return img.astype('float32') / 255.0
