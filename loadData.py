import tensorflow as tf

data_dir = 'Data'


def load_data():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255,
                                                              validation_split=0.2)

    train_dataset = datagen.flow_from_directory(data_dir,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='binary',
                                                subset='training')
    valid_dataset = datagen.flow_from_directory(data_dir,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='binary',
                                                subset='validation')

    return train_dataset, valid_dataset
