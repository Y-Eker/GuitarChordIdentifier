import mediapipe as mp
import cv2
import functions
import numpy as np
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import keras
import tensorflow as tf
from keras import Sequential
from keras.src.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


"""gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.list_physical_devices('GPU')"""
image_extensions = ['jpeg', 'jpg', 'bmp', 'png']
dataset = tf.keras.utils.image_dataset_from_directory("data", labels="inferred")
data_iterator = dataset.as_numpy_iterator()
batch = data_iterator.next()

dataset = dataset.map(lambda x, y: (x / 255, y))
dataset.as_numpy_iterator().next()
data_size = len(np.concatenate([i for x, i in dataset], axis=0))
train_size = int(data_size*.5)
val_size = int(data_size*.25)
test_size = data_size - train_size - val_size

train = dataset.take(train_size)
val = dataset.skip(train_size).take(val_size)
test = dataset.skip(train_size+val_size)

model = Sequential([
    Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(keras.optimizers.Adam(learning_rate=1e-4), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, callbacks=[tensorboard_callback])

pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

webcam = cv2.VideoCapture("videos/GuitarVid4.mp4")
mp_hands = mp.solutions.hands.Hands()
mp_drawing = mp.solutions.drawing_utils
included_landmarks = [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                      mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                      mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                      mp.solutions.hands.HandLandmark.PINKY_TIP]
excluded_landmarks = []

slope_diff_threshold = 0.5
point_diff_threshold = 30

for landmark in mp.solutions.hands.HandLandmark:
    if landmark not in included_landmarks:
        excluded_landmarks.append(landmark)
custom_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
custom_connections = list(mp.solutions.hands.HAND_CONNECTIONS)
for landmark in excluded_landmarks:
    # we change the way the excluded landmarks are drawn
    custom_style[landmark] = DrawingSpec(color=(255, 255, 0), thickness=None)
    # we remove all connections which contain these landmarks
    custom_connections = [connection_tuple for connection_tuple in custom_connections
                          if landmark.value not in connection_tuple]

while webcam.isOpened():
    success, img = webcam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(img)
    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmark, connections=custom_connections,
                                      landmark_drawing_spec=custom_style)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(grayscale_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                   9,15)
    canny_img = cv2.Canny(thresh, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    canny_img = cv2.dilate(canny_img, kernel, iterations=1)
    canny_img = cv2.erode(canny_img, kernel, iterations=1)
    cv2.imshow("Electric Guitar Teacher", canny_img)
    resize = tf.image.resize(canny_img, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    if yhat > 0.5:
        print(f'Am')
    else:
        print(f'C')
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
