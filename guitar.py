import mediapipe as mp
import cv2
import numpy as np
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import keras
import tensorflow as tf


"""gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.list_physical_devices('GPU')"""

model = keras.models.load_model("chord_classifier_model.keras")

webcam = cv2.VideoCapture("videos/GuitarVid5.mp4")
mp_hands = mp.solutions.hands.Hands()
mp_drawing = mp.solutions.drawing_utils
included_landmarks = [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                      mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                      mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                      mp.solutions.hands.HandLandmark.PINKY_TIP]
excluded_landmarks = []

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
                                   9, 15)
    canny_img = cv2.Canny(thresh, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    canny_img = cv2.dilate(canny_img, kernel, iterations=1)
    canny_img = cv2.erode(canny_img, kernel, iterations=1)
    cv2.imshow("Electric Guitar Teacher", img)
    canny_rgb = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2RGB)
    resize = tf.image.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    if yhat > 0.5:
        print(f'Am')
    else:
        print(f'C')
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
