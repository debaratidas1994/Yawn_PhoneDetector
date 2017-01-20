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

prevw=0
prevh=0
prevx=0
prevy=0

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
#track_window2 = (0,0,0,0)
#width = 1920
#height = 1080
threshx,threshy,threshw,threshh=0,0,0,0
areaa=0
width = 1280
height = 960
if not args.get("video", False):
	cap = cv2.VideoCapture(0)
 
# otherwise, grab a reference to the video file
else:
	print 'video input taken'
	cap = cv2.VideoCapture(args["video"])
#fps=float(cap.get(cv2.CAP_PROP_FPS))
fps=30

face_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_mcs_mouth.xml')
ctr=0
flag_thresh=0
q=Queue(maxsize=10)
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),maxLevel = 3,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
finx,finy,finw,finh=0,0,0,0

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
threshold=0

# allow the camera to warmup
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	frame = frame.array
	ctr += 1
	
	#cv2.imshow('Frame111',frame)
	#kwj = cv2.waitKey(60) & 0xff
	#if kwj == 27:
		#rawCapture.truncate(0)
		#break

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,5)       
	if(faces is not None):
		for (x,y,w1,h1) in faces:
			#if((x < width/2) and (y < height/2)):
			if(True):
				
				#print "Face Found"
				#print x,y,w1,h1
				roi_face_gray = gray[y+int(0.6*h1):y+h1,x:x+w1]
				roi_face_colored=frame[y+int(0.6*h1):y+h1,x:x+w1]
				#cv2.imshow('Frameface',roi_face_gray)
				#kwj = cv2.waitKey(60) & 0xff
				#if kwj == 27:
					#break
				mouth_rect  = mouth_cascade.detectMultiScale(roi_face_gray)
				ex,ey,ew,eh=maxarea(mouth_rect)
				if(mouth_rect is not None):
					#mouth_rect=sorted(mouth_rect,reverse=True)[:1]
					#ex,ey,ew,eh=maxarea(mouth_rect)
					#if(ex,ey,ew,eh==0,0,0,0):
						#continue
					#print "mouth Found"
					#if(ctr<=15):
						#if(areaa<ew*eh):
							#threshx,threshy,threshw,threshh=ex,ey,ew,eh
							#areaa=ew*eh
					#else:
						#if(ew*eh<areaa):
							#ex,ey,ew,eh=threshx,threshy,threshw,threshh

					#if(q.full()):
						#dj=q.get()
						#if(ew==0 and eh==0):
							#mouth_rect=dj
							#(ex,ey,ew,eh)=dj
						#q.put(mouth_rect)
					#else:
						#if(ew==0 and eh==0):
							#mouth_rect=q.get()
						#q.put(mouth_rect)
                                        if(ew==0 and eh==0):
                                                print 'Mouth not detected'
                                                ew=50
                                                eh=30
                                                ex=w1/2-25
                                                ey=int(h1*0.4)-30
                                                #ex=prevx
                                                #ey=prevy
                                                #ew=prevw
                                                #eh=prevh
                                                print 'so,ethingignign'
                                        elif(ctr>15):
                                                if(ew*eh<areaa):
                                                        ew=threshw
                                                        eh=threshh
                                                        ex=ex-10
                                                        ey=ey-10
                                        elif(ctr<=15):
                                                if(areaa<ew*eh):
							threshx,threshy,threshw,threshh=ex,ey,ew,eh
							areaa=ew*eh
                                        prevw=ew
                                        prevh=eh
                                        prevx=ex
                                        prevy=ey
                                        if(ey<5):
                                                ey=5
                                        if(ex<5):
                                                ex=5
                                        print 'mouth box'
					print ew,eh
					print 'face box'
					print w1,int(h1*(0.4))
					print roi_face_gray.shape
					#if((ex < w1/2) and (ey > h1/2)):
					if(True):
						roi_mouth_gray=roi_face_gray[ey:ey+eh,ex:ex+ew]
						roi_mouth_colored=roi_face_colored[ey:ey+eh,ex:ex+ew]
					
						#r=x+(ey-20)
						#h=(eh+25)
						#c=y+(ex-25)
						#w=(ew+60)
						ex=ex+x
						eh=(eh)
						ey=ey+y+int(h1*0.6)
						ew=ew
						r0=r
						h0=h
						c0=c
						w0=w
						#if(ctr<=15):
							#if(areaa<ew*eh):
								#threshx,threshy,threshw,threshh=ex,ey,ew,eh
								#areaa=ew*eh
						#else:
							#if(ew*eh<=areaa):
								#ex,ey,ew,eh=threshx,threshy,threshw,threshh
								#ex=ex-10
								#ey=ey-10
								#ew=ew+20
								#eh=eh+20
								#roi_mouth_colored=frame[ey:ey+eh,ex:ex+ew]
								#roi_mouth_gray=gray[ey:ey+eh,ex:ex+ew]
                                                #cv2.imshow('Frameface',roi_mouth_colored)
						#cv2.imshow('Framemouth',roi_mouth_gray)
						#kwj = cv2.waitKey(60) & 0xff
						#if kwj == 27:
							#break
						finx,finy,finw,finh==ex,ey,ew,eh
						hsv_roi =  cv2.cvtColor(roi_mouth_colored, cv2.COLOR_BGR2HSV)
						roi_hist = cv2.calcHist([hsv_roi],[0],None,[256],[0,256])
						cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
						p0 = cv2.goodFeaturesToTrack(roi_mouth_gray,300,0.1,1)
						track_window = (ex,ey,ew,eh)
						#cv2.imshow('Framemouth',roi_mouth_gray)
						#kwj = cv2.waitKey(60) & 0xff
						#if kwj == 27:
							#break

						#cv2.imshow('Frame13',frame)
						#kwj = cv2.waitKey(60) & 0xff
						#if kwj == 27:
							#break
						arr=[]
						std=[]
						total=0
						good_old = np.int0(p0)
						# draw the tracks
						for i in good_old:
							z,v = i.ravel()
							#print str(int(a)+x)+"-----"+str(int(b)+y)
							cv2.circle(frame,(int(z)+ex,int(v)+ey),2,(0,0,255),2)
							arr.insert(0,(z,v))
							std.insert(0,v)
						term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1 )
						for i in range(0,len(arr)):
							for j in range(i+1,len(arr)):
								total +=  np.sqrt((arr[i][0] - arr[j][0])**2 + (arr[i][1] - arr[j][1])**2)
						#if(len(arr)!=0):
						print "no of features-->"+str(len(arr))
						#print  str(total)+"-------"+str(total*1.0/(len(arr)*len(arr)))+"------"+str(np.std(std))
						if(ctr<=15):
                                                        threshold+=np.std(std)
                                                else:
                                                        if(flag_thresh==0):
                                                                threshold=(1.5*threshold)/15
                                                                flag_thresh=1
                                                        print "currnt value->"+str(np.std(std))+"    Threshold value->"+str(threshold)
                                                        if(prevy+eh>(h1*0.4)):
                                                                print 'head turn'
                                                        elif(np.std(std)>threshold):
                                                                print "Yawn detected"
                                                                cv2.putText(frame,'Yawn Dtetced',(10,500),cv2.FONT_HERSHEY_SIMPLEX,4,(0,0,255),2,cv2.LINE_AA)
						cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), 255,2)
						cv2.rectangle(frame, (x,y+int(h1*0.6)), (x+w1,y+h1), 255,2)
						#cv2.rectangle(frame, (finx,finy), (finx+finw,finy+finh), 255,2)
                                                cv2.imshow('Frame13',frame)
                                                rawCapture.truncate(0)
                                                kwj = cv2.waitKey(60) & 0xff
                                                if kwj == 27:
                                                        break
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
		rawCapture.truncate(0)
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

                

	



