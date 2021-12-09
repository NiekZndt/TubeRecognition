import cv2 as cv
import numpy as np

vc = cv.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
    canny = cv.Canny(blur, 0, 175)
    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    blank = np.zeros(frame.shape, dtype='uint8')
    cv.drawContours(blank, contours, -1, (0,0,255), 1)
    key = cv.waitKey(20)

    cv.imshow('camera view', frame)
    cv.imshow('contours', blank)
    if key == 27:
        break

vc.release()
cv.destroyWindow('camera view')
cv.destroyWindow('contours')