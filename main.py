import tensorflow as tf
import numpy as np
import cv2
from loadData import load_data

x_train, y_train, x_valid, y_valid = load_data()

model = tf.keras.Sequential([tf.keras.layers.Input(shape=(27648,)),
                             tf.keras.layers.Dense(units=2048, activation='relu'),
                             tf.keras.layers.Dense(units=1024, activation='relu'),
                             tf.keras.layers.Dense(units=512, activation='relu'),
                             tf.keras.layers.Dense(units=256, activation='relu'),
                             tf.keras.layers.Dense(units=2, activation='sigmoid', name='face_output'),
                             tf.keras.layers.Dense(units=2, activation='sigmoid', name='mask_output')])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss={'face_output': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    'mask_output': tf.keras.losses.BinaryCrossentropy(from_logits=True)},
              metrics={'face_output': 'accuracy', 'mask_output': 'accuracy'})


model.fit(x_train, {'face_output': y_train[:, 0], 'mask_output': y_train[:, 1]},
          validation_data=(x_valid, {'face_output': y_valid[:, 0], 'mask_output': y_valid[:, 1]}),
          epochs=10)

loss, accuracy = model.evaluate(x_valid, {'face_output': y_valid[:, 0], 'mask_output': y_valid[:, 1]})
"""
vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret:
        frame_gray = cv2.resize(frame_gray, (96, 96))
        frame_gray = np.ravel(frame_gray) / 255.0
        frame_gray = np.reshape(frame_gray, (1, 9216))
        prediction = model.predict(frame_gray)
        print(frame_gray.shape)
        prediction_class = np.argmax(prediction, axis=1)
        print("Prediction:", prediction_class)
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
"""
