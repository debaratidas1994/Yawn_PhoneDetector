

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

  pad_w = (int)(0.60*w)
  pad_h = (int)(0.50*h)
  return (max(0,x-pad_w),max(0,y-pad_h),w+2*pad_w,h+2*pad_h)

threshold_talk = 0.12
threshold_no_talk = 0.06
talking = 6
skip_frames = 5
initial_frames = 15
proc_framerate = 5

#During the face intensity detection we define the upper and lower bounds of the array

lower = np.array([0, 48, 20], dtype = "uint8")
upper = np.array([30, 255, 255], dtype = "uint8")

#Misc. variables required for further computation 
hist_sum = []

#~ hands_detected = 0

hands_frames = []
talk_frames = []
curr_talk_frames = []
phone_talk = 0
start_talk = 0
av_face_intensity = 0
threshold_talk_value = 0
threshold_no_talk_value = 0
value_list = []
avg_talk_intensity = 0
curr_value = 0

#camera = cv2.VideoCapture(0)

n = 0

text_display = "Initializing"

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320,240))
 
# allow the camera to warmup
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  #grab the current frame 
  frame = frame.array
  n = n+ 1
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
	  threshold_talk_value = threshold_talk*av_face_intensity + av_face_intensity
	  threshold_no_talk_value = threshold_no_talk*av_face_intensity + av_face_intensity

	  value_list.append(curr_value)

	  current_talk_values = value_list[-talking:]
	  avg_talk_intensity = np.mean(current_talk_values)
	  if phone_talk == 0:
        if avg_talk_intensity >= threshold_talk_value:
		  phone_talk = 1
		  print "Talking"
		  if start_talk == 0:
		    start_talk = 1
		    add_frames = [frame_cnt - i for i in range(0,talking)]
		    add_frames.reverse()
		    curr_talk_frames.extend(add_frames)
		  else:
		    if avg_talk_intensity >= threshold_no_talk_value:
		      curr_talk_frames.append(frame_cnt)
		    else:
			  phone_talk = 0
			  start_talk = 0
			  talk_frames.append(curr_talk_frames)
		      curr_talk_frames = []
		      print "Stop Talking"

          if phone_talk == 1:
            text_display = "Talking on Phone"
        else:
            text_display = "Video Recording"
  cv2.putText(skin,"Threshold Value:" + "%.6f"%threshold_talk_value,(50,200),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2,cv2.LINE_AA)
  cv2.putText(skin,"Current Value:" + "%.6f"%curr_value,(50,250),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2,cv2.LINE_AA)
  cv2.imshow("Frames",np.hstack([mini_frame,skin]))
  print threshold_talk_value," " ,threshold_no_talk_value," ", curr_value," ",avg_talk_intensity

  if phone_talk == 0 and len(talk_frames) != 0:
    cv2.putText(frame,"Last Call:"+ str(len(talk_frames[-1])/2)+" seconds",(100,50),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2,cv2.LINE_AA)

  cv2.putText(frame,text_display,(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2,cv2.LINE_AA)
  ims = cv2.resize(frame,(500,500))
  cv2.namedWindow("Detector")
  cv2.moveWindow("Detector",10,10)
  cv2.imshow("Detector",ims)
  rawCapture.truncate(0)
  key = cv2.waitKey(1) & 0xFF
  

  if key == ord("q"):
    break

cv2.destroyAllWindows()
rawCapture.close()
camera.close()

