import tensorflow as tf
import numpy as np
import cv2
from loadData import load_data

x_train, y_train = load_data()
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(9216,)),
                             tf.keras.layers.Dense(units=2048, activation='relu'),
                             tf.keras.layers.Dense(units=1024, activation='relu'),
                             tf.keras.layers.Dense(units=1024, activation='relu'),
                             tf.keras.layers.Dense(units=128, activation='relu'),
                             tf.keras.layers.Dense(units=2, activation='sigmoid')])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))  # loss and cost

model.fit(x_train, y_train, epochs=10)  # gradient descent
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
