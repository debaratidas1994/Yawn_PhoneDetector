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
#function to get largest conour with key as area
def maxarea(c):
	area=0
	fx,fy,fw,fh=0,0,0,0
	for x,y,w,h in c:
		if(area<w*h):
			fx,fy,fw,fh=x,y,w,h
			area=w*h
	return fx,fy,fw,fh
track_window = (0,0,0,0)

lk_params = dict( winSize  = (15,15),maxLevel = 3,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


prevw=0
prevh=0
prevx=0
prevy=0
#(1-factor) specifies how much percentage of lower face is to be taken
factor=0.65
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",help="path to the (optional) video file")
args = vars(ap.parse_args())
ctr=0
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
x=0
y=0
ex,ey,ew,eh=0,0,0,0
xx,yy,ww,hh=0,0,0,0
flag=0
areaofface=0
#Capture from webcam
if not args.get("video", False):
	cap = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	print 'video input taken'
	cap = cv2.VideoCapture(args["video"])
mouthflag=0
#fps=float(cap.get(cv2.CAP_PROP_FPS))
fps=30

face_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_mcs_mouth.xml')
ctr=0
flag_thresh=0
finx,finy,finw,finh=0,0,0,0

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
threshold=0

# allow the camera to warmup
time.sleep(0.1)
#Capture frame one by one
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	frame = frame.array
	#ctr keeps the frame count
	#gray scale image of the frame
	if(flag==0):
                print 'enter face detection'
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                #Get faces
                faces = face_cascade.detectMultiScale(gray,1.3,5)       
                if(faces is not None):
                        #Get the largest face
                        areaofface=0
                        x,y,w1,h1=0,0,0,0
                        for xx,yy,ww,hh in faces:
                                if(areaofface<ww*hh):
                                        #print xx,yy,ww,hh
                                        areaofface=ww*hh
                                        x,y,w1,h1=xx,yy,ww,hh
                
                        print x,y,w1,h1
                        if(w1!=0 and h1!=0):
                                #take the lower half of the frame, decided by factor value
				roi_face_gray = gray[y+int(factor*h1):y+h1,x:x+w1]
				roi_face_colored=frame[y+int(factor*h1):y+h1,x:x+w1]
				#cv2.imshow('Frameface',roi_face_gray)
				#kwj = cv2.waitKey(60) & 0xff
				#if kwj == 27:
					#break
				#Detect mouth in the lower half face
				mouth_rect  = mouth_cascade.detectMultiScale(roi_face_gray)
				ex,ey,ew,eh=maxarea(mouth_rect)
				if(mouth_rect is not None):
                                        ctr += 1
					ex,ey,ew,eh=maxarea(mouth_rect)
                                        if(ew==0 and eh==0):
                                                #If mouth not found assign the prev frame value
                                                print 'Mouth not detected'
                                                ex=prevx
                                                ey=prevy
                                                ew=prevw
                                                eh=prevh
                                        else:
                                                mouthflag=1
					if(mouthflag==1):
                                                #store the prev frame values
                                                prevw=ew
                                                prevh=eh
                                                prevx=ex
                                                prevy=ey
                                                if(ey<5):
                                                        ey=5
                                                if(ex<5):
                                                        ex=5
                                                #roi_mouth_gray is mouth region in gray scale
						roi_mouth_gray=roi_face_gray[ey:ey+eh,ex:ex+ew]
						#roi_mouth_colored is mouth region in colored form 
						roi_mouth_colored=roi_face_colored[ey:ey+eh,ex:ex+ew]
						ex=ex+x
						eh=(eh)
						ey=ey+y+int(h1*factor)
						ew=ew
						finx,finy,finw,finh==ex,ey,ew,eh
						p0 = cv2.goodFeaturesToTrack(roi_mouth_gray,300,0.1,1)
						track_window = (ex,ey,ew,eh)
						arr=[]
						std=[]
						total=0
                                                if(p0 is not None):
                                                        #print 'enter hua kya/'
                                                        good_old = np.int0(p0)
                                                        # draw the tracks
                                                        for i in good_old:
                                                                z,v = i.ravel()
                                                                #print str(int(a)+x)+"-----"+str(int(b)+y)
                                                                #cv2.circle(frame,(int(z)+ex,int(v)+ey),2,(0,0,255),2)
                                                                arr.insert(0,(z,v))
                                                                std.insert(0,v)
                                                        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1 )
                                                        """for i in range(0,len(arr)):
                                                                for j in range(i+1,len(arr)):
                                                                        total +=  np.sqrt((arr[i][0] - arr[j][0])**2 + (arr[i][1] - arr[j][1])**2)
                                                        #if(len(arr)!=0):
                                                        print "no of features-->"+str(len(arr))
                                                        print  str(total)+"-------"+str(total*1.0/(len(arr)*len(arr)))+"------"+str(np.std(std))
                                                        if(ctr<=15):
                                                                threshold+=np.std(std)
                                                                cv2.putText(frame,'Initializing',(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
                                                        else:
                                                                if(flag_thresh==0):
                                                                        threshold=(1.3*threshold)/15
                                                                        flag_thresh=1
                                                                print "currnt value->"+str(np.std(std))+"    Threshold value->"+str(threshold)
                                                                if(np.std(std)>threshold):
                                                                        print "Yawn detected"
                                                                        cv2.putText(frame,'Yawn Detected',(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
                                                                elif(ctr>15):
                                                                        cv2.putText(frame,'Video Recording',(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
                                                        #cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), 255,2)
                                                        #cv2.rectangle(frame, (x,y+int(h1*factor)), (x+w1,y+h1), 255,2)
                                                        #cv2.rectangle(frame, (finx,finy), (finx+finw,finy+finh), 255,2)
                                                        cv2.namedWindow("Detector")
                                                        cv2.moveWindow("Detector",10,10)
                                                        cv2.imshow('Detector',frame)
                                                        rawCapture.truncate(0)
                                                        kwj = cv2.waitKey(60) & 0xff
                                                        if kwj == 'q':
                                                                break"""
                                                        hsv_roi =  cv2.cvtColor(roi_mouth_colored, cv2.COLOR_BGR2HSV)
                                                        #hsv_roi2 =  cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
                                                        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                                                        roi_hist = cv2.calcHist([hsv_roi],[0],None,[256],[0,256])
                                                        #roi_hist2 = cv2.calcHist([hsv_roi2],[0],None,[256],[0,256])
                                                        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                                                        #cv2.normalize(roi_hist2,roi_hist2,0,255,cv2.NORM_MINMAX)

                                                        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
                                                        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1 )
                                                        flag=1
                                                        temp_old = roi_mouth_gray
                                                        rawCapture.truncate(0)
                                                else:
                                                        rawCapture.truncate(0)
                                                        #when p0 is not detected
                                        else:
                                                rawCapture.truncate(0)
                                                #if mouth is nopt detected
                        else:
                                rawCapture.truncate(0)
                                #if coordinate of face is 0
                else:
                        rawCapture.truncate(0)
                        #if face is not found 
        else:
                print 'enters meanshift-lucas canade'
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,120],1)
		#dst2 = cv2.calcBackProject([hsv],[0],roi_hist2,[0,180],1)

		# apply meanshift to get the new location
		ret, track_window = cv2.meanShift(dst, track_window, term_crit)
		#ret, track_window2 = cv2.meanShift(dst2, track_window2, term_crit)

		# Draw it on image
		x,y,w,h = track_window
		#x2,y2,w2,h2 = track_window2
		if((abs(y - ey) > 100) or (abs(x - ex) > 100)):
			track_window = (ex,ey,ew,eh)
			#flag = 0 
			#continue
		'''if((abs(y2 - r1) > 100) or (abs(x2 - c1) > 100)):
			track_window2 = (c1,r1,w1,h1)
			#flag = 0 
			#continue'''
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		temp = gray[y:y+h,x:x+w]
                p1, st, err = cv2.calcOpticalFlowPyrLK(temp_old, temp, p0, None, **lk_params)
		try:
			arr=[]
			std=[]
			total=0
			# Select good points
			good_new = p1[st==1]
			good_old = p0[st==1]
			# draw the tracks
			for i,(new,old) in enumerate(zip(good_new,good_old)):
				a,b = new.ravel()
				c,d = old.ravel()
				frame = cv2.circle(frame,(int(a)+x,int(b)+y),2,(0,0,255),2)
				arr.insert(0,(a,b))
				std.insert(0,b)
			total=0
			for i in range(0,len(arr)):
				for j in range(i+1,len(arr)):
					total +=  np.sqrt((arr[i][0] - arr[j][0])**2 + (arr[i][1] - arr[j][1])**2)
			if(len(arr)!=0):
				print str(ctr)+"-----"+str(total*1.0/(len(arr)*len(arr)))+"------"+str(np.std(std))

			# Now update the previous frame and previous points
			print 'number of features->'+str(len(arr))
		        temp_old = temp.copy()
		        p0 = good_new.reshape(-1,1,2)
			if(len(good_new) < 10):
				flag=0
		except TypeError:
			print "No points found"
			flag=0
		
		'''corners = cv2.goodFeaturesToTrack(temp,20,0.01,10)
		corners = np.int0(corners)
		for i in corners:
			l,k = i.ravel()
			frame = cv2.circle(frame,(l+x,k+y),2,(0,0,255),2)'''
		img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
		#img2 = cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), 255,2)
                cv2.imshow('img2',img2)
                rawCapture.truncate(0)
                kwj = cv2.waitKey(60) & 0xff
                if kwj == 'q':
                        break
                rawCapture.truncate(0)
	
