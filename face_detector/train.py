import os
from xml.etree.ElementInclude import FatalIncludeError
from PIL import Image
import numpy as np
import cv2
import pickle




x_train= []
detecti= cv2.CascadeClassifier('frontalface.xml')
recognizer= cv2.face.LBPHFaceRecognizer_create()

#base directory path(finding current file). dirname gives errything upto file in thw path for parameter
BASE_DIR= os.path.dirname(os.path.abspath(__file__))

#path to img folder concatenated
image_dir= os.path.join(BASE_DIR, "img")
labels={}
c_id= 0

y_labels= []
x_train= []
#laying all the files. file gives file name and root gives path till that folder
for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg"):
            path= os.path.join(root, file)
            label= os.path.basename(os.path.dirname(path))
            if label not in labels:
                labels[label]= c_id
                c_id += 1
            id= labels[label]
            pil_image= Image.open(path).convert("L")
            size= (550,550)
            final= pil_image.resize(size, Image.ANTIALIAS)
            print(path)
            

            
            #converts img to numbers
            image_array= np.array(final, "uint8")
            print(image_array)
            #print(image_array)
            face= detecti.detectMultiScale(image_array)
            for (x,y,h,w) in face:
                roi= image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id)
#dumping labels dictionary in file for later use
with open("labels.pickle", 'wb') as f:
    pickle.dump(labels, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")