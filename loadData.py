import tensorflow as tf

train_dir = 'Data/train'
valid_dir = 'Data/validation'


def load_data():
    train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    validation = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    train_dataset = train.flow_from_directory(train_dir,
                                              target_size=(224, 224),
                                              batch_size=20,
                                              class_mode='binary')
    valid_dataset = validation.flow_from_directory(valid_dir,
                                                   target_size=(224, 224),
                                                   batch_size=20,
                                                   class_mode='binary')
    return train_dataset, valid_dataset
