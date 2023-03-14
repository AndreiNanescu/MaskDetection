import tensorflow as tf
from loadData import load_data

train_set, valid_set = load_data()

model = tf.keras.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                             tf.keras.layers.MaxPool2D(2, 2),

                             tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                             tf.keras.layers.MaxPool2D(2, 2),

                             tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                             tf.keras.layers.MaxPool2D(2, 2),

                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(512, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')
                             ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model = model.fit(train_set, validation_data=valid_set, epochs=10)

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
