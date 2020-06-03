import cv2
import numpy as np
import time
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
face_without_mask = cv2.CascadeClassifier('src\haarcascade_mcs_mouth.xml')
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    temp = False
    faces = face_without_mask.detectMultiScale(gray, 2.8, 18)
    lenth = len(faces)
    if lenth != 0:
        face_with_mask = cv2.CascadeClassifier('src\haarcascade_frontalface_default.xml')
        faces_with = face_with_mask.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces_with:
            org = (x,y)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255),3)
            cv2.putText(img,"NOT WEARING MASK",org,cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),4,False)
            crop_img = img[y:y+h, x:x+w]
            cv2.imwrite('write_images/without/'+str(x)+'.jpg',crop_img)
    if lenth == 0: 
        face_with_mask = cv2.CascadeClassifier('src\haarcascade_frontalface_default.xml')
        faces_with = face_with_mask.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces_with:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0),3)
            org = (x,y)
            cv2.putText(img,"WEARING MASK",org,cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0),4,False)
            crop_img = img[y:y+h, x:x+w]
            cv2.imwrite('write_images/with/'+str(x)+'.jpg',crop_img)
    cv2.imshow('frame',frame)
 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()