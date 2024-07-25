# Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame
import os
import smtplib
import threading
import requests

# Get the username of the current user

userName = os.getlogin()

# Define constants

ALARM_ON = False
model_path = "shape_predictor_68_face_landmarks.dat"
ALARM_PATH = "assets_alarm.mp3"


# Define a function to play the alarm sound
def sound():
    """
    Play the alarm sound using pygame.
    """
    pygame.mixer.init()
    pygame.mixer.music.load('assets_alarm.mp3')
    pygame.mixer.music.play()
    pygame.time.Clock().tick(10)


def eye_aspect_ratio(eye):
    """
    Calculate the eye aspect ratio.

    Parameters:
    eye_coordinates (list): List of eye coordinates.

    Returns:
    float: Eye aspect ratio.
    """
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def email(text):
    """
    Send an email using smtplib.

    Parameters:
    email_body (str): Email body.
    """
    while True:
        try:
            requests.get("https://www.google.com")
            break
        except:
            os.system("shutdown /s /t 0")
    send = "mhsn97864@gmail.com"
    receive = "hdhdhsjsg4@gmail.com"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(send, "whvv sdtw gcgb ohkx")
    server.sendmail(send, receive, text)


def detect():
    """
    Detect drowsiness using facial landmarks.
    """
    global ALARM_ON
    
    thresh = 0.20
    
    frame_check = 15
    
    detects = dlib.get_frontal_face_detector()
    
    predict = dlib.shape_predictor(model_path)
    
    lStart, lEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    
    rStart, rEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    
    cap = cv2.VideoCapture(0)
    
    flag = 0
    
    nonPerson = 0
    
    minutes = 0
    while True:
        # Read a frame from the camera
        frame = cap.read()[1]
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=640, height=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame

        subjects = detects(gray, 0)

        # If no face is detected, display a message
    
        if not subjects:
            nonPerson += 1
        else:
            # If a face is detected, reset the non-person counter
            if nonPerson > 60:
                message = f"{userName} is far for pc! \n\n time :{nonPerson / 60}m"
                t1 = threading.Thread(target=email, args=(message,))
                t1.start()
                t1.join()
            nonPerson = 0
            
            # Loop through each face detected
            
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                # Check if the eye
                if ear < thresh:
                    
                    flag += 1
                   
                    print("ear: ", flag)
                    
                    if flag >= frame_check:
                        
                        if not ALARM_ON:
                            ALARM_ON = True

                        if flag == 60:
                            
                            flag = 0
                            
                            minutes += 1
                        print("Drowsy")
                        th = threading.Thread(target=sound)
                        th.start()
                        th.join()
                else:
                    if flag > 30 or minutes >= 1:
                   
                        message = f"{userName} is Drowsy! \n\n time :{flag} S  {minutes} M"
                        
                        t2 = threading.Thread(target=email, args=(message,))
                        
                        t2.start()
                        
                        t2.join()
                    
                    flag = 0
                   
                    minutes=0
                    
                    ALARM_ON = False
                    
        #cv2.imshow("frame",frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    detect()
