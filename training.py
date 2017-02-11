import cv2
import glob
import random
import numpy as np
import time
import pickle 
import requests

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list

fishface = cv2.createFisherFaceRecognizer() #Initialize fisher face classifier
# mapping = [0,-1,-1,-1,-1,1,-1,1]


def run_recognizer():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)
     
    while True:
        faces = ()
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=10,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        
        facePresent = 0
        if(len(faces)>0):
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                gray = gray[y:y+h, x:x+w] #Cut the frame to size
                try:

                    gray = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                    prediction_data=(gray)

                except:
                   # print("hi")
                    print("lul\n")
            pred, conf = fishface.predict(prediction_data)
            r = requests.post('192.168.20.20:1234', data = {'result':pred, 'conf':conf})

  

fishface.load("trained_classifier")
run_recognizer()

# Thanks to van Gent, P. (2016). Emotion Recognition With Python, OpenCV and a Face Dataset. A tech blog about fun things with Python and embedded electronics. Retrieved from:
# http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/