

from picamera.array import PiRGBArray
from picamera import PiCamera
import time

from os import listdir
from os.path import isfile, join
import numpy
import cv2
import imutils
import numpy as np
import argparse
import glob
from itertools import groupby
from operator import itemgetter
import sys

#argument parser to add argument for giving fps being used

ap = argparse.ArgumentParser()

ap.add_argument('-f','--fps',help = "fps to process")

args = vars(ap.parse_args())

#fps variable specifying the fps given by the user
fps = int(args["fps"])

def extract_features(image):
    face_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # iterate over all identified faces and try to find eyes
    return faces
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

def pad(x,y,w,h):

  pad_w = (int)(0.55*w)
  pad_h = (int)(0.40*h)
  return (max(0,x-pad_w),max(0,y-pad_h),w+2*pad_w,h+3*pad_h)

if fps == 10:
  threshold = 0.28
  stitch = 5
  talking = 4
  skip_frames = 5
  initial_frames = 15
  proc_framerate = 5
elif fps == 1:
  threshold = 0.35
  stitch = 3
  talking = 4
  initial_frames = 5
else:
  threshold = 0.45
  stitch = 15
  talking = 30
  initial_frames = 6

#During the face intensity detection we define the upper and lower bounds of the array

lower = np.array([0, 48, 20], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

#Misc. variables required for further computation 

hist_sum = []

hands_detected = 0

hands_frames = []
talk_frames = []
curr_talk_frames = []
phone_talk = 0
start_talk = 0
av_face_intensity = 0
threshold_hand = 0
temp_stitch = 0

#camera = cv2.VideoCapture(0)

n = 0

text_display = "Video Recording"

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  #grab the current frame 
  frame = frame.array
  n = n+ 1

  if n % proc_framerate == 0:

    frame_cnt = n/proc_framerate
    if frame_cnt > skip_frames:

      if frame_cnt <= initial_frames:
        my_face = extract_features(frame)
        if len(my_face) != 0:
          x,y,w,h = my_face[0]
          x,y,w,h = pad(x,y,w,h)
          
      mini_frame = frame[y:y+h,x:x+w]  
      converted = cv2.cvtColor(mini_frame,cv2.COLOR_BGR2HSV)
      skinMask = cv2.inRange(converted,lower,upper)
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
      skinMask = cv2.erode(skinMask,kernel,iterations = 2)
      skinMask = cv2.dilate(skinMask,kernel,iterations = 2)
      skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
      skin = cv2.bitwise_and(mini_frame,mini_frame,mask = skinMask)
      hist_mask = cv2.calcHist([mini_frame],[0],skinMask,[256],[0,256])
      curr_value = np.sum(hist_mask)
      if frame_cnt <= initial_frames:
        hist_sum.append(curr_value)
      else:
        hist_sum.sort()
        av_face_intensity = np.mean(hist_sum[3:8])
        threshold_hand = threshold*av_face_intensity + av_face_intensity

        if curr_value > threshold_hand:
          hands_detected += 1
          hands_frames.append(frame_cnt)
          if hands_detected == talking:
            phone_talk = 1
            print "talking"
            if start_talk == 0:
              start_talk = 1
              curr_talk_frames.extend(hands_frames[-talking:])
              
      
        else:
          if phone_talk == 1:
            temp_stitch += 1
            if temp_stitch > stitch:
              phone_talk = 0
              print "stop talking"
              start_talk = 0
              hands_detected = 0
              del curr_talk_frames[-stitch:]
              talk_frames.append(curr_talk_frames) 
              curr_talk_frames = []
          else:
            hands_detected = 0

        if phone_talk == 1:
          curr_talk_frames.append(frame_cnt)
          text_display = "Talking on Phone"

      cv2.putText(skin,"Threshold Value:" + "%.6f"%threshold_hand,(50,200),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2,cv2.LINE_AA)
      cv2.putText(skin,"Current Value:" + "%.6f"%curr_value,(50,250),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2,cv2.LINE_AA)
      cv2.imshow("Frames",np.hstack([mini_frame,skin]))
      print threshold_hand," " , curr_value

  if phone_talk == 0 and len(talk_frames) != 0:
    cv2.putText(frame,"Last Call:"+ str(len(talk_frames[-1]))+" seconds",(200,300),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2,cv2.LINE_AA)

  cv2.putText(frame,text_display,(300,400),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2,cv2.LINE_AA)
  #cv2.imshow("toShow",frame)
  rawCapture.truncate(0)
  key = cv2.waitKey(1) & 0xFF
  

  if key == ord("q"):
    break

camera.release()
cv2.destroyAllWindows()
