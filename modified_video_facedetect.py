# import the necessary packages

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from os import listdir
from os.path import isfile, join
import imutils
import numpy as np
import argparse
import glob
from itertools import groupby
from operator import itemgetter
import sys
from time import sleep

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
    cv2.waitKey(1)
    #cv2.destroyAllWindows()



# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
	image = imutils.resize(image,width = 400,height = 600)
        extract_features(image)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
