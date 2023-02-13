import os
import pathlib
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from matplotlib import rcParams
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False


def process_image(img_path: str) -> np.array:
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img = img.resize(size=(96, 96))
    img = np.ravel(img) / 255.0
    return img


def process_folder(folder: pathlib.PosixPath) -> pd.DataFrame:
    # We'll store the images here
    processed = []

    # For every image in the directory
    for img in folder.iterdir():
        # Ensure JPG
        if img.suffix == '.jpg' or img.suffix == '.jpeg' or img.suffix == '.png':
            # Two images failed for whatever reason, so let's just ignore them
            try:
                processed.append(process_image(img_path=str(img)))
            except Exception as _:
                continue

    # Convert to pd.DataFrame
    processed = pd.DataFrame(processed)
    # Add a class column - face or a mask
    processed['class'] = folder.parts[-1]

    return processed


def load_data():
    train_face = process_folder(folder=pathlib.Path.cwd().joinpath('Data/Face'))
    train_mask = process_folder(folder=pathlib.Path.cwd().joinpath('Data/Mask'))
    train_set = pd.concat([train_face, train_mask], axis=0)

    with open('train_set.pkl', 'wb') as f:
        pickle.dump(train_set, f)

    x_train = train_set.drop('class', axis=1)
    y_train = train_set['class']
    y_train.factorize()
    y_train = tf.keras.utils.to_categorical(y_train.factorize()[0], num_classes=2)
    x_train = x_train.values
    return x_train, y_train
