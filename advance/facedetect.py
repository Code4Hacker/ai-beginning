import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image.flags.writeable = True
    
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(20,140,34), thickness=1, circle_radius=1),mp_drawing.DrawingSpec(color=(33,120,14), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(45,140,44), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(53,130,24), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(45,140,44), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(53,130,24), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(45,140,44), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(53,130,24), thickness=2, circle_radius=2))
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
    while True:
        success, frame = cap.read()
        if success == True:
        
            image, results = mediapipe_detection(frame, holistic)
            
            draw_landmarks(image, results)
            cv2.imshow("Video Face Detector", image) 
        else:
            print("error")
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()