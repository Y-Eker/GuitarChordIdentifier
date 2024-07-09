import mediapipe as mp
import cv2
import numpy as np
from mediapipe.python.solutions.drawing_utils import DrawingSpec


def main():
    webcam = cv2.VideoCapture(0)
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
    print(excluded_landmarks)
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
        canny_img = cv2.Canny(grayscale_img, 125, 175)
        lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 150)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * -b)
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * -b)
            y2 = int(y0 - 1000 * a)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("Electric Guitar Teacher", img)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
