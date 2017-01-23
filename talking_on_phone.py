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

from picamera.array import PiRGBArray
from picamera import PiCamera
import time

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

  return faces

def pad(x,y,w,h):
  pad_w = (int)(0.50*w)
  pad_h = (int)(0.50*h)
  
  return (max(0,x-pad_w),max(0,y-pad_h),w+2*pad_w,h+2*pad_h)

threshold_talk = 0.18
threshold_no_talk = 0.10
talking = 8
skip_frames = 5
initial_frames = 15
proc_framerate = 1

#During the face intensity detection we define the upper and lower bounds of the array

lower = np.array([0, 48, 20], dtype = "uint8")
upper = np.array([30, 255, 255], dtype = "uint8")

#Misc. variables required for further computation 

hist_sum = []


talk_frames = []
curr_talk_frames = []
phone_talk = 0
start_talk = 0
av_face_intensity = 0
value_list = []
current_talk_values = []
avg_talk_intensity = 0
threshold_talk_value = 0
threshold_no_talk_value = 0
frame_area = 0
#~ reInitialize = 0
#~ cnt = 0
skin = []
x,y,w,h = 0,0,0,0

# initialize the camera and grab a reference to the raw camera capture


n = 0

text_display = "Initializing"

camera = PiCamera()
camera.resolution = (480, 360)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(480,360))

time.sleep(0.1)

frame_cnt = 0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  
  frame = frame.array
  n = n+ 1

  if n % proc_framerate == 0:

    frame_cnt += 1
    if frame_cnt > skip_frames:

	  my_face = extract_features(frame)
	  frame_area = 0
	  if len(my_face) == 0:
		frame_cnt -= 1
		text_display = "Face not Detected,look into the camera"
		
		cv2.putText(frame,text_display,(40,60),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 255), 2,cv2.LINE_AA)
		im = cv2.resize(frame,(600,600))
		cv2.namedWindow("toShow",flags = cv2.WINDOW_NORMAL)
		cv2.moveWindow("toShow",200,200)
		cv2.imshow("toShow",im)
		
		key = cv2.waitKey(1) & 0xFF
		rawCapture.truncate(0)
		if key == ord("q"):
		  break
		continue
	  for faces in my_face:
	    x_temp,y_temp,w_temp,h_temp = faces
	    if frame_area < w_temp*h_temp:
	      x,y,w,h = faces
	      frame_area = w*h 
	  x,y,w,h = pad(x,y,w,h)
	  mini_frame = frame[y:y+h,x:x+w]
	  frame_area = w*h  
	  converted = cv2.cvtColor(mini_frame,cv2.COLOR_BGR2HSV)
	  skinMask = cv2.inRange(converted,lower,upper)
	  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
	  skinMask = cv2.erode(skinMask,kernel,iterations = 3)
	  skinMask = cv2.dilate(skinMask,kernel,iterations = 2)
	  skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
	  skin = cv2.bitwise_and(mini_frame,mini_frame,mask = skinMask)
	  hist_mask = cv2.calcHist([mini_frame],[0],skinMask,[256],[0,256])
	  curr_value = np.sum(hist_mask)/frame_area
	  
	  if frame_cnt <= initial_frames:
	    hist_sum.append(curr_value)
      #elif reInitialize == 1:
        #text_display = "Reinitializing"
        #print "Reinitializing"
        #if(cnt == 0):
          #hist_sum = []
        #cnt += 1
        #hist_sum.append(curr_value)
        #if cnt == 10:
          #reInitialize = 0
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
	        print "Talking"
	        phone_talk = 1
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
		  text_display = ""
      
	  print threshold_talk_value," ",threshold_no_talk_value," ",curr_value," ", avg_talk_intensity
	  cv2.imshow("Frames",np.hstack([mini_frame,skin]))

  if phone_talk == 0 and len(talk_frames) != 0:
    cv2.putText(frame,"Last Call: About "+ str(len(talk_frames[-1])/2)+" seconds",(140,30),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2,cv2.LINE_AA)

  cv2.putText(frame,text_display,(60,60),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2,cv2.LINE_AA)
  im = cv2.resize(frame,(600,600))
  cv2.namedWindow("toShow",flags = cv2.WINDOW_NORMAL)
  cv2.moveWindow("toShow",200,200)
  cv2.imshow("toShow",im)



  key = cv2.waitKey(1) & 0xFF
  rawCapture.truncate(0)

  if key == ord("q"):
    break

camera.close()
cv2.destroyAllWindows()



