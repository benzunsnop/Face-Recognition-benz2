import cv2,os
import numpy as np
from PIL import Image 
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('trainer/trainer.yml');
cascadePath = ("haarcascade_frontalface_default.xml");
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataSet'

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
id=0
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        if(id==1):
             id="Benz"
        elif(id==2):
             id="Obama"
        elif(id==3):
             id="Bas"
        #cv2.cv.PutText(cv2.cv.fromarray(im),str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 255) #Draw the text
        cv2.putText(im, str(id), (x,y+h),font, 1, (0, 0, 255), 2)
        cv2.imshow('im',im)
    if(cv2.waitKey(1)==ord('q')):
     break;
cam.release()
cv2.destroyAllWindows()

