import tensorflow as tf
from loadData import load_data
from plotting import plot_data
train_set, valid_set, test_set = load_data()

model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3),
                                                    kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                             tf.keras.layers.MaxPool2D(3, 3),
                             tf.keras.layers.Dropout(0.35),

                             tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                                    kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                             tf.keras.layers.MaxPool2D(3, 3),
                             tf.keras.layers.Dropout(0.35),

                             tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                                    kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                             tf.keras.layers.MaxPool2D(3, 3),
                             tf.keras.layers.Dropout(0.35),

                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(128, activation='relu',
                                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                             tf.keras.layers.Dropout(0.50),
                             tf.keras.layers.Dense(64, activation='relu'),
                             tf.keras.layers.Dropout(0.50),
                             tf.keras.layers.Dense(1, activation='sigmoid')
                             ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_set, steps_per_epoch=train_set.samples // 32, validation_data=valid_set,
                    validation_steps=valid_set.samples // 32, epochs=30)

print(model.predict(test_set))
plot_data(history)
