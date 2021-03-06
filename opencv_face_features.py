#!/usr/bin/python2.7

# Sample script for demonstrating face detection with OpenCV on a 
# Raspberry Pi. The script takes an image as input parameter or makes
# a snapshot with the Pi camera module when no input is provided.
# Then, the image is searched for faces and the faces found are examined
# for eyes. The results are displayed graphically; therefore, the script needs
# a running desktop environment.

# See the tutorial
# http://www.opencv-primer-face-detection-with-the-raspberry-pi
# for more info!

import sys
from time import sleep
import cv2
import picamera
import imutils

def extract_features(image):
    face_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/Nariz.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # iterate over all identified faces and try to find eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, minSize=(30, 30))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

        noses = nose_cascade.detectMultiScale(roi_gray, minSize=(125, 30))
        for (ex,ey,ew,eh) in noses:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)


    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) >= 2:
        image_file = sys.argv[1]
    else:
        sleep(2)
        image_file = 'snapshot.jpg'
        picamera.PiCamera().capture(image_file)

    image = cv2.imread(image_file)
    image=imutils.resize(image,width=600,height=480)
    extract_features(image)
