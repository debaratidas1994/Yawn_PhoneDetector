# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

from os import listdir
from os.path import isfile, join

# ---------------------
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

if fps == 10:
  threshold = 0.45
  stitch = 5
  talking = 5
  initial_frames = 6
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

hist_sum = 0

hands_detected = 0

hands_frames = []
talk_frames = []
curr_talk_frames = []
phone_talk = 0
start_talk = 0
av_face_intensity = 0
threshold_hand = 0
temp_stitch = 0

# ---------------------

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320,240))
 
# allow the camera to warmup
time.sleep(0.1)

n=0 

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  #grab the current frame 
  n = n+ 1
  if n % 1!= 0:
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)

    if key == ord("q"):
      break
    continue

  frame = frame.array
  #frame = frame[0:300,0:350]
  frame = imutils.resize(frame,width = 500,height = 600)
  converted = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
  skinMask = cv2.inRange(converted,lower,upper)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
  skinMask = cv2.erode(skinMask,kernel,iterations = 2)
  skinMask = cv2.dilate(skinMask,kernel,iterations = 2)
  skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
  skin = cv2.bitwise_and(frame,frame,mask = skinMask)
  #mini_frame = frame[:,100:400]
  #mini_skin = skinMask[:,100:400]
  hist_mask = cv2.calcHist([frame],[0],skinMask,[256],[0,256])
  #hist_mask = cv2.calcHist([mini_frame],[0],mini_skin,[256],[0,256])


  #Misc. Text
  text_display = "Video Recording"

  #Function call to calculate face_intensity should replace the code here
  curr_value = np.sum(hist_mask)
  frame_cnt = n
  if frame_cnt <= initial_frames:
    hist_sum += curr_value
  else:
    av_face_intensity = hist_sum/initial_frames
    threshold_hand = threshold*av_face_intensity + av_face_intensity
    cv2.putText(skin,"Threshold Value:" + "%.6f"%threshold_hand,(150,200),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2,cv2.LINE_AA)
    
    #detect hand
    if curr_value > threshold_hand:
      hands_detected += 1
      temp_stitch = 0
      hands_frames.append(frame_cnt)
      if hands_detected == talking:
        phone_talk = 1
        if start_talk == 0:
          start_talk = 1
          curr_talk_frames.extend(hands_frames[-talking:])
        
    else:
      if phone_talk == 1:
        temp_stitch += 1
        if temp_stitch > stitch:
          phone_talk = 0
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

  cv2.putText(frame,text_display,(300,250),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2,cv2.LINE_AA)
  print text_display , "Threshold " , threshold_hand , "Current ",curr_value
  if phone_talk == 0 and len(talk_frames) != 0:
    cv2.putText(frame,"Last Call:"+ str(len(talk_frames[-1]))+" seconds",(200,200),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2,cv2.LINE_AA)

  cv2.putText(skin,"Current Value:" + "%.6f"%curr_value,(150,250),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2,cv2.LINE_AA)
  cv2.imshow("images",frame)
  #cv2.imshow("miniIMages",np.hstack([mini_frame,mini_skin]))

  #print "hist mask at ",n ," = " ,np.sum(hist_mask)
  key = cv2.waitKey(1) & 0xFF
  
  rawCapture.truncate(0)

  if key == ord("q"):
    break



  

