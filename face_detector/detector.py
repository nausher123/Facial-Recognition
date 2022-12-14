import cv2
import pickle
from datetime import date
import time
from os.path import exists
import os.path



today= date.today()

name= f"attendance {today}.txt"
recognizer= cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
#captures videp the brackets can have a video too
oldlabels= {}
x=1

#load the label dictionary
with open("labels.pickle", "rb") as f:
    oldlabels= pickle.load(f)
    labels= {v:k for k,v in oldlabels.items()}
    print(labels)
image= cv2.imread("suii.jpg")
#loops over frams
while True:
#creates the detector frontal face is the training data classifier is what makes it recogninse
    detecti= cv2.CascadeClassifier('frontalface.xml')

    #reads the current frame
    

    #cv2.imshow('window', img)

    grey= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #detects objects and returns coordinates of rectangles, in this case detects faces cos our data is of faces
    coordinates= detecti.detectMultiScale(grey)
    
    for (x,y,w,h) in coordinates:
        roi_grey= grey[y:y+h, x: x+w]
    #draw rectangle
        cv2.rectangle(image, (x,y), (x+h, y+w), (255,0,0), 2)
        #returns id and confidence level
        id, conf= recognizer.predict(roi_grey)
        if conf>= 80: 
           message= f"{labels[id]} is present. date:{today}"
           print(message)
        cv2.putText(image, str(labels[id]), (20,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 128))
        present= labels[id]
        
    
    
    
       
            

    cv2.imshow('window', image)
    
    key= cv2.waitKey(1)
    time.sleep(7)
    break
import pathlib
file = pathlib.Path(name)
if file.exists ():
    f= open(name, "a")
    f.writelines(f"\n {message}")
    f.close()
else:
    f= open(name, "x")
    f.writelines(f"\n {message}")
    f.close()