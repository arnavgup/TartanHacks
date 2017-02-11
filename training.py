import cv2
import glob
import random
import numpy as np
import time
import pickle 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import os
import random
from pykeyboard import PyKeyboard

class makeBrowser(object):

    def __init__(self, x):
        self.current_url = x

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list

fishface = cv2.createFisherFaceRecognizer() #Initialize fisher face classifier
# mapping = [0,-1,-1,-1,-1,1,-1,1]

def getVideo():
    vids=[]
    vids.append(("/watch?v=dQw4w9WgXcQ&t=0m43s",17))
    vids.append(("/watch?v=ZZ5LpwO-An4",12))
    vids.append(("/watch?v=76_03o9bEPQ&t=0m35s",23))
    return vids[random.randrange(0,len(vids))]


def run_recognizer():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)

    # ret, frame = video_capture.read()
    # cv2.imshow('Video', frame)
    count=10
    while True:
        # Capture frame-by-frame
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
            print(pred)

            if(pred in {1,2,3,4,6} and count>20):
                count=0
                webURL = "https://youtube.com"
                webAdd,wtime = getVideo()
                print("play vid")

                # chop = webdriver.ChromeOptions()
                # chop.add_extension('Adblock-Plus_v1.12.4.crx')
                # driver = webdriver.Chrome("./chromedriver",chrome_options = chop)
                driver = webdriver.Chrome("./chromedriver")
                driver.get(webURL+webAdd)
                # driver.find_element_by_tag_name('body').send_keys(Keys.COMMAND + Keys.TAB)
                # driver.switch_to.window(driver.window_handles[-1])
                # first_link = first_result.find_element_by_tag_name('a')
                # first_link.send_keys(Keys.CONTROL + '1')
                
                # driver.find_element_by_tag_name('body').send_keys(Keys.COMMAND + 't') 
                time.sleep(1)
                # k = PyKeyboard()
                # k.press_key('F')
                # youtubePlayer = driver.find_element_by_id("page-container") 
                # # youtubePlayer = driver.find_element_by_id("player-api")
                # youtubePlayer.send_keys('F')
                time.sleep(wtime-1)
                driver.close()
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            w,h=video_capture.get(3),video_capture.get(4)
            finalImage=makeThug(orig,faces,mouths,eyes)
            makeVideo(finalImage,w,h,'p')
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count+=1



    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # print(gray)
        # faces = faceCascade.detectMultiScale(
        #     gray,
        #     scaleFactor=1.2,
        #     minNeighbors=10,
        #     minSize=(30, 30),
        #     flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        # )
        
        # cv2.imshow('Video', frame)
       

  

fishface.load("trained_classifier")

run_recognizer()
# Thanks to van Gent, P. (2016). Emotion Recognition With Python, OpenCV and a Face Dataset. A tech blog about fun things with Python and embedded electronics. Retrieved from:
# http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/