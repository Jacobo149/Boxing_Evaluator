import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def recognize_gesture(landmarks):
    """
    Recognize basic gestures based on hand landmarks.
    """
    # Get the coordinates of the thumb tip (landmark 4), index tip (landmark 8), etc.
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Gesture: Open Hand (fingers spread apart)
    if index_tip.x > thumb_tip.x and middle_tip.x > index_tip.x and ring_tip.x > middle_tip.x and pinky_tip.x > ring_tip.x:
        return "Open Hand"

    # Gesture: Fist (fingers curled in)
    if abs(index_tip.x - thumb_tip.x) < 0.02 and abs(middle_tip.x - index_tip.x) < 0.02 and abs(ring_tip.x - middle_tip.x) < 0.02 and abs(pinky_tip.x - ring_tip.x) < 0.02:
        return "Fist"

    return "Unknown"


while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize the gesture from the hand landmarks
            gesture = recognize_gesture(hand_landmarks.landmark)
            cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
