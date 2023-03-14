import pathlib
import numpy as np
import pandas as pd
import sklearn.utils
import tensorflow as tf
import cv2

train_dir = 'Data/train'
valid_dir = 'Data/valid'

train_dataset = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255).flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
valid_dataset = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255).flow_from_directory(
    train_dir,
    target_size=(
        224, 224),
    batch_size=32,
    class_mode='binary'
)
train_dataset.class_indices
valid_dataset.class_indices
"""""
def process_image(img_path: str) -> np.array:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96))
    img = np.ravel(img) / 255.0
    return img


def process_folder(folder: pathlib.PosixPath) -> pd.DataFrame:
    processed = []
    for img in folder.iterdir():
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
    train_set = sklearn.utils.shuffle(train_set)

    x_train = train_set.drop('class', axis=1)
    y_train = train_set['class']
    y_train_factorized = y_train.factorize()[0]
    y_train = tf.keras.utils.to_categorical(y_train_factorized, num_classes=2)

    valid_face = process_folder(folder=pathlib.Path.cwd().joinpath('Data/FaceValid'))
    valid_mask = process_folder(folder=pathlib.Path.cwd().joinpath('Data/MaskValid'))
    valid_set = pd.concat([valid_face, valid_mask], axis=0)
    valid_set = sklearn.utils.shuffle(valid_set)

    x_valid = valid_set.drop('class', axis=1)
    y_valid = valid_set['class']
    y_valid_factorized = y_valid.factorize()[0]
    y_valid = tf.keras.utils.to_categorical(y_valid_factorized, num_classes=2)

    return x_train, y_train, x_valid, y_valid
"""""
