import mediapipe as mp
import cv2
import time
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer =  mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        print(results)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h,w,c = image.shape
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h

                
            
                for num,hand in enumerate(hand_landmarks.landmark):
                    cx,cy = int(hand.x * w), int(hand.y * h)
                    print(f"landmark{num}: {cx},{cy}")
                    if cx > x_max:
                        x_max = cx
                    if cx < x_min:
                        x_min = cx
                    if cy > y_max:
                        y_max = cy
                    if cy < y_min:
                        y_min = cy
                padding = 20
                cv2.rectangle(image, 
                            (x_min - padding, y_min - padding), 
                            (x_max + padding, y_max + padding), 
                            (0, 255, 0), 2)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
                 


        #display the image
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
# model_path ='C:/Users/USER/mediapipe-autoscroll/gesture_recognizer.task'
# base_options = BaseOptions(model_asset_path=model_path)

def get_angle(a,b,c):
    vecBA_x = a.x - b.x
    vecBA_y = a.y - b.x
    vecBC_x = c.x - b.x
    vecBC_y = c.y - c.x

    mag_BA = math.sqrt(vecBA_x**2 + vecBA_y**2)
    mag_BC = math.sqrt(vecBC_x**2 + vecBC_y**2)

    dot_product = (vecBC_x * vecBC_x) + (vecBA_y * vecBC_y)

    if mag_BA == 0 or mag_BC == 0:
        return 0
    
    cos_theta = max(-1.0, min(1.0, cos_theta))
    cos_theta = dot_product / (mag_BC * mag_BA)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg
