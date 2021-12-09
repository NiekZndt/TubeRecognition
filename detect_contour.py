import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def find_high_low_pixels(image):
    h = image.shape[0]
    w = image.shape[1]
    high = loop_over_pixels(range(0, h), range(0, w), image)
    low = loop_over_pixels(range(h, 0), range(w, 0), image)
    #if(not high or not low):
        #print('Error, no high low values')
    #else:
        #print(high, low, low)
    print(high)
    image[high[0], high[1]] = [0, 255, 0]

    print(low)
    image[low[0], low[1]] = [0, 255, 0]
    cv.imshow('Remake', image)
    return high

def loop_over_pixels(vertical_range, horizontal_range, image):
    for x in horizontal_range:
        for y in vertical_range:
            #print(image[y,x], y, x)
            if (image[y,x,2] == 255):
                return [x,y]
    return False

img = cv.imread('Images/tube7.jpg')

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

#find_high_low_pixels(blank)

cv.waitKey(0)