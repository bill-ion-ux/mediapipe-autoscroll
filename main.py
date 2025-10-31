import mediapipe as mp
import cv2
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    #display the image
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
# model_path ='C:/Users/USER/mediapipe-autoscroll/gesture_recognizer.task'
# base_options = BaseOptions(model_asset_path=model_path)
