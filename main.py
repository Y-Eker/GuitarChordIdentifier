import mediapipe as mp
import cv2


def main():
    webcam = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    while webcam.isOpened():
        success, img = webcam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_hands.Hands().process(img)
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmark, connections=mp_hands.HAND_CONNECTIONS)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Electric Guitar Teacher", img)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
