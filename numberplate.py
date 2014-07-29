#!/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt

filepath = "/Users/andy/work/headon/_M2B3091.jpg"

img = cv2.imread(filepath,0)
blur_img = cv2.blur(img,(5,5))


#sobel_img = cv2.Sobel(blur_img,cv2.CV_64F,1,0,3,1,0)
sobel_img = cv2.Sobel(blur_img,-1,1,0)

ret,threshold_img = cv2.threshold(sobel_img,0,255,cv2.THRESH_OTSU)

element = cv2.getStructuringElement(cv2.MORPH_RECT,(44,6))
morphed_img = cv2.morphologyEx(threshold_img,cv2.MORPH_CLOSE,element)
#

#kernel = np.ones((8,8),np.uint8)
morphed_img2 = morphed_img.copy()
#morphed_img2 = cv2.dilate(morphed_img,kernel)
#morphed_img2 = cv2.morphologyEx(threshold_img,cv2.MORPH_CLOSE,kernel)
contours, hierarchy = cv2.findContours(morphed_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#plt.imshow(cont_img, cmap = 'gray')

def verifySizes(mr):
  error = 0.4
  aspect = 4.7272
  minArea = 31*aspect*31
  maxArea = 125*aspect*125
  ratioMin = aspect-aspect*error
  ratioMax = aspect+aspect*error

  # example - gives cooridinates, size, angle..
# ((1271.2064208984375, 1051.0308837890625), (23.04122543334961, 2.588901996612549), -21.250507354736328)
  width = mr[1][0]
  height = mr[1][1]
  if any([height == 0, width == 0]):
    return False

  area = width * height
  aspectRatio = float(width) / float(height)
  if(aspectRatio<1):
    aspectRatio = 1/aspectRatio
  if any([area < minArea, area > maxArea, aspectRatio < ratioMin, aspectRatio > ratioMax]):
    return False
  return True

print "before ", len(contours)

rects = []
newcontours = []

for contour in contours:
  mr = cv2.minAreaRect(contour)
  if (verifySizes(mr)):
    rects.append(mr)
    newcontours.append(contour)

print "after ", len(rects)
print len(newcontours)


cv2.drawContours(morphed_img,newcontours,-1,(128,255,255),3)

cv2.imshow('contours', morphed_img)
#plt.subplot(2,1,1)
#plt.imshow(morphed_img2)
#plt.subplot(2,1,2),
#plt.imshow(morphed_img)
#plt.show()
#cv2.imshow('image',sobel_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
