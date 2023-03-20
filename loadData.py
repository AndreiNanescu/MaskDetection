import tensorflow as tf

data_dir = 'Data'
test_data_dir = 'testData'


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

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    test_dataset = test_datagen.flow_from_directory(test_data_dir, target_size=(224, 224), class_mode='binary',
                                                    subset='training')

    return train_dataset, valid_dataset, test_dataset
