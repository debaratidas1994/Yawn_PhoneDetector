#!/usr/bin/python
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

import numpy as np
import cv2
import sys
import argparse
import imutils
from Queue import *

def __init__(self):
	pass

def maxarea(c):
	area=0
	fx,fy,fw,fh=0,0,0,0
	for x,y,w,h in c:
		if(area<w*h):
			fx,fy,fw,fh=x,y,w,h
			area=w*h
	return fx,fy,fw,fh



ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",help="path to the (optional) video file")
args = vars(ap.parse_args())
ctr=0
flag=0
flag2=0
r=0
h=0
c=0
w=0
r0=0
h0=0
c0=0
w0=0
r1=0
h1=0
c1=0
w1=0
track_window = (0,0,0,0)
width = 1280
height = 960
face_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_mcs_mouth.xml')

ctr=0
q=Queue(maxsize=10)
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),maxLevel = 3,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
finx,finy,finw,finh=0,0,0,0


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 

# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	frame=frame.array
	ctr += 1
	cv2.imshow('Frame111',frame)
	kwj = cv2.waitKey(60) & 0xff
	if kwj == 27:
		rawCapture.truncate(0)
		break
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	print "ok"
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	if(faces is not None):
		for (x,y,w1,h1) in faces:
			if(True):
				roi_face_gray = gray[y+int(0.6*h1):y+h1,x:x+w1]
				roi_face_colored=frame[y+int(0.6*h1):y+h1,x:x+w1]
				cv2.imshow('Frameface',roi_face_gray)
				kwj = cv2.waitKey(60) & 0xff
				if kwj == 27:
						break
				mouth_rect  = mouth_cascade.detectMultiScale(roi_face_gray)
				ex,ey,ew,eh=maxarea(mouth_rect)
				if(mouth_rect is not None):                          
					if(q.full()):
							dj=q.get()
							if(ew==0 and eh==0):
									mouth_rect=dj
									(ex,ey,ew,eh)=dj
							q.put(mouth_rect)
					else:
							if(ew==0 and eh==0):
									mouth_rect=q.get()
							q.put(mouth_rect)
					print ew,eh
                                           
					if(True):
							roi_mouth_gray=roi_face_gray[ey:ey+eh,ex:ex+ew]
							roi_mouth_colored=roi_face_colored[ey:ey+eh,ex:ex+ew]
							cv2.imshow('Framemouth',roi_mouth_gray)
							kwj = cv2.waitKey(60) & 0xff
							if kwj == 27:
									break
					
							ex=ex+x
							eh=(eh)
							ey=ey+y+int(h1*0.6)
							ew=ew
							r0=r
							h0=h
							c0=c
							w0=w
							finx,finy,finw,finh==ex,ey,ew,eh
							hsv_roi =  cv2.cvtColor(roi_mouth_colored, cv2.COLOR_BGR2HSV)
							roi_hist = cv2.calcHist([hsv_roi],[0],None,[256],[0,256])
							cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
							p0 = cv2.goodFeaturesToTrack(roi_mouth_gray,300,0.1,1)
							track_window = (ex,ey,ew,eh)
							cv2.imshow('Framemouth',roi_mouth_gray)
							kwj = cv2.waitKey(60) & 0xff
							if kwj == 27:
									break

                                                   
							arr=[]
							std=[]
							total=0
							good_old = np.int0(p0)
							# draw the tracks
							for i in good_old:
									z,v = i.ravel()
									cv2.circle(frame,(int(z)+ex,int(v)+ey),2,(0,0,255),2)
									arr.insert(0,(z,v))
									std.insert(0,v)
							term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1 )
							for i in range(0,len(arr)):
									for j in range(i+1,len(arr)):
											total +=  np.sqrt((arr[i][0] - arr[j][0])**2 + (arr[i][1] - arr[j][1])**2)
							print  str(total)+"-------"+str(total*1.0/(len(arr)*len(arr)))+"------"+str(np.std(std))
							cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), 255,2)
							cv2.rectangle(frame, (x,y+int(h1*0.6)), (x+w1,y+h1), 255,2)
							#print 'width height ratio'
							#print float(ew)/eh
							#img2 = cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), 255,2)
							#cv2.namedWindow('img2',cv2.WINDOW_NORMAL)
							#if(img2!=None):
							#cv2.imshow('img2',img2)
							#k = cv2.waitKey(60) & 0xff
							#if k == 27:
							#	break
							#out.write(img2)
#else:

	#cv2.imwrite(chr(k)+".jpg",img2)
	else:
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,120],1)
			#dst2 = cv2.calcBackProject([hsv],[0],roi_hist2,[0,180],1)

			# apply meanshift to get the new location
			ret, track_window = cv2.CamShift(dst, track_window, term_crit)
			#ret, track_window2 = cv2.meanShift(dst2, track_window2, term_crit)

			# Draw it on image
			ex,ey,ew,eh = track_window
			finx,finy,finw,finh=track_window
			roi_mouth_colored=frame[ey:ey+eh,ex:ex+ew]
			roi_hist=roi_mouth_colored
			print 'yoyo'

	cv2.rectangle(frame, (finx,finy), (finx+finw,finy+finh), 255,2)
	cv2.imshow('Frame13',frame)
	rawCapture.truncate(0)
	kwj = cv2.waitKey(60) & 0xff
	if kwj == 27:
			break




