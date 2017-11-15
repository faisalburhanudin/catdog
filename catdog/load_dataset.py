import os

import numpy as np
from PIL import Image


def resize_image(path, sz=(256, 256)):
    """Resize image and return numpy array

    :arg path: image path
    :arg sz: image new shape

    :return numpy array with shape (256, 256, 3)
    """
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert('RGB')
    im = im.resize(sz, resample=Image.BILINEAR)
    im = np.asarray(im, dtype=np.uint8)
    return im


def to_categorical(file):
    return [1, 0] if file[:3] == "cat" else [0, 1]


def data_generator(batch=32):
    path = '/home/faisal/catdog/data/train'
    files = os.listdir(path)

    total_loop = int(len(files) / batch)
    while True:
        for i in range(total_loop):
            file = files[i * batch:(i + 1) * 32]

            images = np.array([resize_image(os.path.join(path, i), sz=(32, 32)) for i in file])
            labels = np.array(list(map(to_categorical, file)))
            yield images, labels
