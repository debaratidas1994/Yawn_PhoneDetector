import cv2
import sys
import numpy as np

import dlib
flag=0
total=0
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
predictor_path ="shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
frame_no=0
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    frame_no+=1

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
         # Converting the OpenCV rectangle coordinates to Dlib rectangle
        # Converting the OpenCV rectangle coordinates to Dlib rectangle
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        #print dlib_rect

        detected_landmarks = predictor(frame, dlib_rect).parts()

        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        # copying the image so we can see side-by-side
        image_copy = frame.copy()
        maxy=0
        miny=1000
        arr=[]
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            

            # annotate the positions
            if(idx>47):
                #print pos
                cv2.putText(image_copy, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
                maxy=max(maxy,pos[1])
                miny=min(miny,pos[1])

           

                #cv2.putText(image_copy, str(idx), pos,
                 #       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                  #      fontScale=0.4,ld 
                   #     color=(0, 0, 255))

            # draw points on the landmark positions
            #   cv2.circle(image_copy, pos, 3, color=(0, 255, 255))

        #print maxy-miny
    # Display the resulting frame
        if frame_no<15:
            total+=(maxy-miny)
            print str(maxy-miny)
        else:
            if(flag==0):
                total=total/15
                flag=1
                total=total*1.5
            print 'current value----->'+str(maxy-miny)+"    threshold value "+str(total)
            if((maxy-miny)>total):
                cv2.putText(image_copy,'yawn detected',(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                print 'yawn'
    #cv2.imshow('Video', frame)
    cv2.imshow("Landmarks found", image_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()