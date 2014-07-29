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

#plt.matshow(cont_img)
print contours


cv2.drawContours(morphed_img,contours,-1,(128,255,255),3)

cv2.imshow('contours', morphed_img)
#plt.subplot(2,1,1)
#plt.imshow(morphed_img2)
#plt.subplot(2,1,2),
#plt.imshow(morphed_img)
#plt.show()
#cv2.imshow('image',sobel_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
