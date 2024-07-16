import mediapipe as mp
import cv2
import functions
import numpy as np
from mediapipe.python.solutions.drawing_utils import DrawingSpec


def main():
    webcam = cv2.VideoCapture("videos/GuitarVid2.mp4")
    mp_hands = mp.solutions.hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    included_landmarks = [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                          mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                          mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                          mp.solutions.hands.HandLandmark.PINKY_TIP]
    excluded_landmarks = []

    slope_diff_threshold = 2
    point_diff_threshold = 20

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
        thresh = cv2.threshold(grayscale_img, 150, 255, cv2.THRESH_BINARY)[1]
        canny_img = cv2.Canny(thresh, 100, 200)
        lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 150, np.array([]), 20, 75)
        line_image = np.copy(img) * 0
        strong_lines = []
        if lines is not None:
            for line in lines:
                if functions.similar_strong_line(line, strong_lines, slope_diff_threshold, point_diff_threshold):
                    continue
                if functions.calculate_slope(line) > 4:
                    continue
                strong_lines.append(line)
            for line in strong_lines:
                x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]
                # noinspection PyTypeChecker
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        cv2.imshow("Electric Guitar Teacher", img)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
