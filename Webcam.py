import cv2
import os
import logging as log
import datetime as dt
from time import sleep
import numpy as np
import tensorflow
from keras.models import load_model
model = load_model('masknet.h5')
print(dt.datetime.now())

def detect(im):
    try:
        im=cv2.resize(im,(128,128))
    except:
        pass
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    prediction = np.argmax(prediction)
    return prediction
try:  
    if not os.path.exists('D:\dataset\mask_on'): 
        os.makedirs('D:\dataset\mask_on')
except OSError: 
    print('OS Error')
try:  
    if not os.path.exists('D:\dataset\mask_off'): 
        os.makedirs('D:\dataset\mask_off')
except OSError: 
    print('OS Error')

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)
mask_on_count=0
mask_off_count=0
video_capture = cv2.VideoCapture(1)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to open webcam!')
        sleep(5)
        break

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20,20)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 2)

    #if anterior != len(faces):
        # anterior = len(faces)
        # log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)
    if len(faces)>=1:
        EXTENSION = 'png'
        file_name_format = "{:%Y%m%d_%H%M%S}_{:d}.{:s}"
        date = dt.datetime.now()
        for (x,y,w,h) in faces:
            crop_img=frame[y-5:y+h+5,x-5:x+w+5,]
            i=detect(crop_img)
            if i==0:                
                mask_on_count+=1
                filename = file_name_format.format(date, mask_on_count, EXTENSION)
                crop_img=cv2.resize(crop_img,(256,256))
                #cv2.imwrite('D:\dataset\mask_on\ '+filename,crop_img)
                log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
            else:
                mask_off_count+=1
                filename = file_name_format.format(date, mask_off_count, EXTENSION)
                crop_img=cv2.resize(crop_img,(256,256)) 
                cv2.imwrite('D:\dataset\mask_off\ '+filename,crop_img)        
        cv2.imshow('Video', frame)
        sleep(5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
print(dt.datetime.now())
cv2.destroyAllWindows()
