import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('Images/tube1.jpg')

resized_image = rescaleFrame(img, .2)
cv.imshow('Resized', resized_image)

blank = np.zeros(resized_image.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(resized_image , cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

canny = cv.Canny(blur, 0, 175)
cv.imshow('Canny edges', canny)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)