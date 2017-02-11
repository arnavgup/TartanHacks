import cv2
import glob
import random
import numpy as np
import time
import pickle 

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
# emotions = ["neutral", "anger", "happy", "sadness", "surprise"] #Emotion list

fishface = cv2.createFisherFaceRecognizer() #Initialize fisher face classifier
# mapping = [0,-1,-1,-1,-1,1,-1,1]
data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []

    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))

    return training_data, training_labels

def run_recognizer(training_data, training_labels):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    video_capture = cv2.VideoCapture(0)
    # # Capture frame-by-frame
    # ret, frame = video_capture.read()
    # # for processing
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = ()
    
    while (len(faces)==0):
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)
    # for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=10,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    
    # Draw a rectangle around the faces
    prediction_data = []
    prediction_labels = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        gray = gray[y:y+h, x:x+w] #Cut the frame to size
        try:
            # print(emotion, filenumber)
            gray = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
            # print("size fixed \n",gray)
        except:
           # print("hi")
            print("lul\n")#If error, pass file
        prediction_data.append(gray)
        prediction_labels.append(1)
    cv2.imwrite("output.jpeg",gray)
    
    # video_capture.release()
    # cv2.destroyAllWindows()
    print "training fisher face classifier"
    print "size of training set is:", len(training_labels), "images"


    print "predicting classification set"
    cnt = 0
    correct = 0
    incorrect = 0
    pred, conf = fishface.predict(gray)
    print(pred, prediction_labels[cnt])
    #     if pred == prediction_labels[cnt]:
    #         correct += 1
    #         cnt += 1
    #     else:
    #         incorrect += 1
    #         cnt += 1
    # return ((100*correct)/(correct + incorrect))

#Now run it
# metascore = []
# # for i in range(0,2):
# correct = run_recognizer()
# print "got", correct, "percent correct!"
# metascore.append(correct)

# print "\n\nend score:", np.mean(metascore), "percent correct!"
training_data, training_labels= make_sets()
# print(type(training_data),type(training_labels))
# with open("data", 'wb') as f:
#     pickle.dump(training_data, f)
# with open("labels", 'wb') as f:
#     pickle.dump(training_labels, f)
with open("data", 'rb') as f:
    training_data = pickle.load(f)
with open("labels", 'rb') as f:
    training_labels = pickle.load(f)   
# with open("labels", 'rb') as f:
#     training_labels = pickle.load(f)   
fishface.train(training_data, np.asarray(training_labels))
fishface.save("trained_classifier")
# fishface.load("trained_classifier")
print("wutface")
# with open("trained", 'wb') as f:
#     pickle.dump(x, f)

# print((training_data),(training_labels)) 
while True:
    print("gg")
    run_recognizer(training_data, training_labels)
    # time.sleep(2)
# while True:
#     time.sleep(5)
#     run_recognizer(training_data, training_labels)
#     if cv2.waitKey(1) & 0xFF == ord('p'):
#         w,h=video_capture.get(3),video_capture.get(4)
#         # finalImage=makeThug(orig,faces,mouths,eyes)
#         # makeVideo(finalImage,w,h,'p')
#         break

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
