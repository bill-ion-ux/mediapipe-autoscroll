import mediapipe as mp
import cv2
import time
import numpy as np
import math
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer =  mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
last_scroll_time = 0
SCROLL_COOLDOWN = 0.5

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:
    def get_angle(a,b,c):
        vecBA_x = a.x*w - b.x*w
        vecBA_y = a.y*h - b.y*h
        vecBC_x = c.x*w - b.x*w
        vecBC_y = c.y*h - b.y*h

        mag_BA = math.sqrt(vecBA_x**2 + vecBA_y**2)
        mag_BC = math.sqrt(vecBC_x**2 + vecBC_y**2)

        dot_product = (vecBA_x * vecBC_x) + (vecBA_y * vecBC_y) 

        if mag_BA == 0 or mag_BC == 0:
            return 0
        
        cos_theta = dot_product / (mag_BC * mag_BA)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_rad = math.acos(cos_theta)
        angle_deg = angle_rad * (180 / math.pi)
        return angle_deg
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
                pointing_up = False
                pointing_down = False
                angle_index_pip = get_angle(hand_landmarks.landmark[5],hand_landmarks.landmark[6],hand_landmarks.landmark[7])
                
                
                is_index_straight = angle_index_pip > 140
                is_middle_bent = get_angle(hand_landmarks.landmark[9],hand_landmarks.landmark[10],hand_landmarks.landmark[12]) < 110
                is_ring_bent = get_angle(hand_landmarks.landmark[13],hand_landmarks.landmark[14],hand_landmarks.landmark[16]) < 110
                is_pinky_bent = get_angle(hand_landmarks.landmark[17],hand_landmarks.landmark[18],hand_landmarks.landmark[20]) < 110  
                
                is_pointing_up_direction = hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y
                is_pointing_down_direction = hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y
                all_others_bend = is_middle_bent and is_pinky_bent and is_ring_bent
                if all_others_bend and is_index_straight:
                    if is_pointing_up_direction:
                        pointing_up = True
                        gesture_text = "Pointing UP"
                       
                    elif is_pointing_down_direction:
                        pointing_down = True
                        gesture_text = "Pointing DOWN"
                        
                else:
                    gesture_text = "none"

                for num,hand in enumerate(hand_landmarks.landmark):
                    cx,cy = int(hand.x * w), int(hand.y * h)
                    # print(f"landmark{num}: {cx},{cy}")
                    if cx > x_max:
                        x_max = cx
                    if cx < x_min:
                        x_min = cx
                    if cy > y_max:
                        y_max = cy
                    if cy < y_min:
                        y_min = cy
                padding = 20
                current_time = time.time() 

   
                if (current_time - last_scroll_time) > SCROLL_COOLDOWN:
                    if pointing_up:
                        pyautogui.scroll(100) 
                        last_scroll_time = current_time
                    elif pointing_down:
                        pyautogui.scroll(-100) # Scroll DOWN
                        last_scroll_time = current_time

                    
                cv2.rectangle(image, 
                            (x_min - padding, y_min - padding), 
                            (x_max + padding, y_max + padding), 
                            (0, 255, 0), 2)
                cv2.putText(image, gesture_text, 
                    (x_min - padding, y_min - padding - 10), # Position (10px above the box)
                    cv2.FONT_HERSHEY_SIMPLEX, # Font
                    0.9, # Font scale
                    (0, 255, 0), # Color (Green)
                    2) # Thickness
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

