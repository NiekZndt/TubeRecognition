import cv2 as cv
import numpy as np

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result


path = input('Enter an image name: ')
img = cv.imread('Images/' + path)
width = int(img.shape[1] * 30 / 100)
height = int(img.shape[0] * 30 / 100)
dim = (width, height)
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
l, a, b = cv.split(lab)
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv.merge((cl,a,b))
final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
#frame = frame[300: 630, 150:390]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 0, 175)

blank = np.zeros(img.shape, dtype='uint8')
#cv.rectangle(blank,(300,100),(660,440),(0,255,0),1)

key = cv.waitKey(3)
#print(contours)

angle = 0
angleSmallestSurface = 0
smallestSurface = False
while angle != 360:
    angle += 1
    key = cv.waitKey(1)
    contours, hierarchies = cv.findContours(rotate_image(canny, angle), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    blank = np.zeros(img.shape, dtype='uint8')
    cv.drawContours(blank, contours, -1, (0,0,255), 1)

    largestItem = (0, 0, 0, 0)
    for c in contours:
        # get bounding rect
        (x,y,w,h) = cv.boundingRect(c)
        if(w * h > largestItem[2] * largestItem[3]):
            largestItem = (x, y, w, h)

    if(smallestSurface == False):
        angleSmallestSurface = angle
        smallestSurface = largestItem
    if smallestSurface[1] * smallestSurface[2] > largestItem[1] * largestItem[2]:
        angleSmallestSurface = angle
        smallestSurface = largestItem

    cv.rectangle(blank, (largestItem[0],largestItem[1]), (largestItem[0]+largestItem[2],largestItem[1]+largestItem[3]), (0, 255, 0), 2)
    cv.imshow('Blank', blank)
    cv.imshow('Rotation', rotate_image(img, angle))

    # if key == 27:
    #     break

cv.destroyWindow('Blank')
cv.destroyWindow('Rotation')
canny = rotate_image(canny, -angleSmallestSurface)
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
blank = np.zeros(img.shape, dtype='uint8')

largestItem = (0, 0, 0, 0)
for c in contours:
    # get bounding rect
    (x,y,w,h) = cv.boundingRect(c)
    if(w * h > largestItem[2] * largestItem[3]):
        largestItem = (x, y, w, h)
    #draw red rect
    #cv.rectangle(blank, (x,y), (x+w,y+h), (0, 0, 255), 2)
    
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.rectangle(blank, (largestItem[0],largestItem[1]), (largestItem[0]+largestItem[2],largestItem[1]+largestItem[3]), (0, 255, 0), 2)
cv.imshow('camera view', img)
cv.imshow('contours', rotate_image(blank, angleSmallestSurface))
cv.imshow('Rotated view', rotate_image(img, angleSmallestSurface))
print('Width | Height: ', largestItem[2], largestItem[3])

cv.waitKey(0)

cv.destroyWindow('camera view')
cv.destroyWindow('contours')