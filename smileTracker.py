import numpy as np
import cv2
import math as m
from time import time




#####################
##                functions                  ##
#####################
# returning the absolute angle between two objects
def ang(obj):
    return abs(m.atan((obj[0][1]-obj[1][1])/\
                                       (obj[0][0]-obj[1][0]))*180/m.pi)

# return the distance between two objects
def dist(obj):
    return ((obj[0][0]-obj[1][0])**2 + (obj[0][1]-obj[1][1])**2)**0.5

# color selector
def selectColor(symetry):
    global good_angle, bad_angle
    if symetry[0][0] == symetry[1][0]:
        return bad
    else:
        angle = ang(symetry)
        if angle <= good_angle:
            return good
        elif angle <= bad_angle:
            return medium
        else:
            return bad

# void callback for trackers
def clbk(val):
    pass
#             END of functions              #
## _____________________ ##




#####################
##         video defenition              ##
#####################
#-- locating the first active camera, open it and start capturing
cap = cv2.VideoCapture(0)
capturing = False # for activating capture photo of smile
framing = False # for showing framing on face

#-- cascade data for face and eye detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

#-- face tilt angles
good_angle = 1.5
bad_angle = 4.5

#-- color defenition
good = (0,255,0)
medium = (0, 255, 255)
bad = (0,0,255)
nothing = (0,0,0)

#-- Trackbar window and Trackers  --#

#--  eyes and nose section
# window
wind = 'eyes and nose'
cv2.namedWindow(wind)
cv2.resizeWindow(wind, 500,160)
# scalars
scaleFactorVal = 102 # acctually 1.02
minNeighborsVal = 26
minSizeVal = 41
maxSizeVal = 90
# trackers
cv2.createTrackbar('scaleFactorVal', wind, scaleFactorVal, 140, clbk)
cv2.createTrackbar('minNeighborsVal', wind, minNeighborsVal, 45, clbk)
cv2.createTrackbar('minSizeVal', wind, minSizeVal, 80, clbk)
cv2.createTrackbar('maxSizeVal', wind, maxSizeVal, 220, clbk)

#--  smile section
# window
wind2 = 'smile'
cv2.namedWindow(wind2)
cv2.resizeWindow(wind2, 500,160)
# scalars
scaleFactorVal2 = 140 # acctually 1.40
minNeighborsVal2 = 29
minSizeVal2 = 47
maxSizeVal2 = 100
# trackers
cv2.createTrackbar('scaleFactorVal2', wind2, scaleFactorVal2, 140, clbk)
cv2.createTrackbar('minNeighborsVal2', wind2, minNeighborsVal2, 45, clbk)
cv2.createTrackbar('minSizeVal2', wind2, minSizeVal2, 80, clbk)
cv2.createTrackbar('maxSizeVal2', wind2, maxSizeVal2, 220, clbk)

#            END of defenition             #
## _____________________ ##




#####################
##              mouse event               ##
#####################
def click(event, x, y, flags, param):
        global framing, capturing
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= y <= 70:
            if 0 <= x <= 110:
                capturing = not capturing
            elif img.shape[1]-170 <= x <= img.shape[1]:
                framing = not framing
        
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click)
#          END of mouse event            #
## _____________________ ##



#####################
##                main loop                  ##
##               img analyz                 ##
#####################
while True:

    #-- Tracker controler for eyes and nose
    sFVal = cv2.getTrackbarPos("scaleFactorVal", wind)/100
    mNVal = cv2.getTrackbarPos("minNeighborsVal", wind)
    mnSVal = cv2.getTrackbarPos("minSizeVal", wind)
    mxSVal = cv2.getTrackbarPos("maxSizeVal", wind)

    #-- Tracker controler for smile
    sFVal2 = cv2.getTrackbarPos("scaleFactorVal2", wind2)/100
    mNVal2 = cv2.getTrackbarPos("minNeighborsVal2", wind2)
    mnSVal2 = cv2.getTrackbarPos("minSizeVal2", wind2)
    mxSVal2 = cv2.getTrackbarPos("maxSizeVal2", wind2)

    #-- snapshot of this current frame
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    #-- turning to gray and sprad gray vals more evenly by histograma
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    #-- locating faces in the image and analyze the detected section
    faces = face_cascade.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        if framing:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,0),1)
        faceROI = gray[y:y+h,x:x+w] # croping face area from gray img

    
        #-- detect eyes in each face
        eyes = eye_cascade.detectMultiScale(faceROI,\
                                        scaleFactor=sFVal, minNeighbors=mNVal,\
                                            minSize=(mnSVal,mnSVal), maxSize=(mxSVal,mxSVal))
        eyes_centers = [] # for indicating of nose position
        if len(eyes) == 2:
            clr = selectColor(eyes) # pick the color by the tilt of the eyes
            angle = ang(eyes) # for presenting the angle on the screen
            #-- text for face angle
            cv2.putText(img, "face angle", org = (img.shape[1]//2-110,30),
                        fontFace = cv2.FONT_HERSHEY_COMPLEX,
                        fontScale = 1.1,
                        color = (0,0,0),
                        thickness = 2)
            cv2.putText(img, str(round(angle,2)), org = (img.shape[1]//2-110,65),
                        fontFace = cv2.FONT_HERSHEY_COMPLEX,
                        fontScale = 1.1,
                        color = clr,
                        thickness = 2)
            # finde inner frame to minimize area for latter detacting of smile and nose
            # and wrap if wraping is active
            for (x2, y2, w2, h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                eyes_centers.append(eye_center)
                if framing:
                    radius = int(round((w2 + h2)*0.25))
                    cv2.circle(img, eye_center, radius, clr, 1)

                    
            #-- detect nose in each face
            # y -> from eye to end of face
            # x -> from left eye to right eye
            eyes_centers.sort() # for placing the left eye first
            left_eye_X = eyes_centers[0][0]
            right_eye_X = eyes_centers[1][0]
            top = eyes_centers[0][1]
            bottom = y + h
            lowwerFaceROI = gray[top:bottom, left_eye_X:right_eye_X]


            #-- detect nose in each face
            nose = nose_cascade.detectMultiScale(lowwerFaceROI,\
                                            scaleFactor=sFVal, minNeighbors=mNVal,\
                                            minSize=(mnSVal,mnSVal), maxSize=(mxSVal,mxSVal))
            if len(nose) == 1:
                x2, y2, w2, h2 = nose[0]
                nose_center = (left_eye_X + x2 + w2//2, top + y2 + h2//2)
                if framing:
                    radius = int(round((w2 + h2)*0.25))
                    cv2.circle(img, nose_center, radius, (255,255,0), 6)
                    # OPTIONAL for framing the lips
                    cv2.rectangle(img,
                                  (left_eye_X, nose_center[1]+h2//3),
                                  (right_eye_X, nose_center[1]+4*h2//3),
                                  (200,140,170),
                                  2)


        #-- detect smile in each face   :)
        smileROI = lowwerFaceROI if len(eyes) == 2 else faceROI
        smile = smile_cascade.detectMultiScale(faceROI,\
                                        scaleFactor=sFVal2, minNeighbors=mNVal2,\
                                            minSize=(mnSVal2,mnSVal2), maxSize=(mxSVal2,mxSVal2))
        ## case of detecting each lips as seperate mouth
        if len(smile) == 2:
            lips, curve = smile[0], smile[1]
            if lips[0] == curve[0] or 75 <= ang([lips, curve]):
                smile_center = (x + lips[0] + w2//2, y + lips[1] + h2//2)
                if framing:
                    radius = int(round((w2 + h2)*0.25))
                    cv2.circle(img, smile_center, radius, (0,64,0), 3)
        ## case of detecting normal mouth with two lips
        else:
            for (x2, y2, w2, h2) in smile:
                smile_center = (x + x2 + w2//2, y + y2 + h2//2)
                if framing:
                    radius = int(round((w2 + h2)*0.25))
                    cv2.circle(img, smile_center, radius, (255,255,255), 3)


            #-- take a snapshot of smile if it's active and deactivating
            if capturing and len(smile):
                cv2.imwrite("./photos/smile_" + str(round(time()%1000,2)) + '.jpg', img)
                capturing = False

    ##   TEXT SECTION ##
    #-- text for smile
    cv2.putText(img, "smile", org = (3,30),
                fontFace = cv2.FONT_HERSHEY_COMPLEX,
                fontScale = 1.1,
                color = (0,0,0),
                thickness = 2)
    cv2.putText(img, "ON" if capturing else "OFF", org = (3,65),
                fontFace = cv2.FONT_HERSHEY_COMPLEX,
                fontScale = 1.1,
                color = (0,255,0) if capturing else (0,0,255),
                thickness = 2)

    #-- text for framing
    cv2.putText(img, "framing", org = (img.shape[1]-170,30),
                fontFace = cv2.FONT_HERSHEY_COMPLEX,
                fontScale = 1.1,
                color = (0,0,0),
                thickness = 2)
    cv2.putText(img, "ON" if framing else "OFF", org = (img.shape[1]-170,65),
                fontFace = cv2.FONT_HERSHEY_COMPLEX,
                fontScale = 1.1,
                color = (0,255,0) if framing else (0,0,255),
                thickness = 2)

    #-- present the current frame
    cv2.imshow("Image",img)

    #-- frame each 3 milisecond, q for exit
    ch = cv2.waitKey(3)
    if ch & 0xFF == ord('q'):
        break
#            END of main loop               #
## _____________________ ##



# end of video
cap.release()
cv2.destroyAllWindows()













