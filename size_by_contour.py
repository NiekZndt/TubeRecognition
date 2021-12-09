import cv2 as cv
import numpy as np
import time

vc = cv.VideoCapture(0)
# url = 'http://192.168.178.235:8080/video'
# vc = cv.VideoCapture(url)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    frame = cv.resize(frame, (960, 540), interpolation = cv.INTER_AREA)
    #frame = frame[300: 630, 150:390]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
    canny = cv.Canny(blur, 0, 150)
    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    blank = np.zeros(frame.shape, dtype='uint8')
    #cv.rectangle(blank,(300,100),(660,440),(0,255,0),1)

    key = cv.waitKey(3)
    #print(contours)
    largestItem = (0, 0, 0, 0)

    i = 0
    angle = 0
    while i < 360:
        i += 1
        image_center = tuple(np.array(canny.shape[1::-1]) / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, 1, 1.0)
        canny = cv.warpAffine(canny, rot_mat, canny.shape[1::-1], flags=cv.INTER_LINEAR)
        contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        for c in contours:
            # get bounding rect
            (x,y,w,h) = cv.boundingRect(c)
            if(w * h > largestItem[2] * largestItem[3]):
                largestItem = (x, y, w, h)
                angle = i
            # draw red rect
            #cv.rectangle(blank, (x,y), (x+w,y+h), (0, 0, 255), 2)
    
    cv.drawContours(blank, contours, -1, (0,0,255), 1)
    cv.rectangle(blank, (largestItem[0],largestItem[1]), (largestItem[0]+largestItem[2],largestItem[1]+largestItem[3]), (0, 255, 0), 2)
    cv.imshow('camera view', frame)
    cv.imshow('contours', blank)
    if key == 27:
        break


vc.release()
cv.destroyWindow('camera view')
cv.destroyWindow('contours')