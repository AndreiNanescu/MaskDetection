import pathlib
import numpy as np
import pandas as pd
import sklearn.utils
import tensorflow as tf
import cv2

train_dir = 'Data/train'
valid_dir = 'Data/validation'

train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
validation = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

train_dataset = train.flow_from_directory(train_dir,
                                          target_size=(224, 224),
                                          batch_size=20,
                                          class_mode='binary'
                                          )
valid_dataset = validation.flow_from_directory(valid_dir,
                                               target_size=(224, 224),
                                               batch_size=20,
                                               class_mode='binary'
                                               )
train_dataset.class_indices
valid_dataset.class_indices

