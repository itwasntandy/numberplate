#!/usr/bin/python

import sys
sys.path.append('/usr/local/opencv2/lib/python2.7/site-packages')
import cv2
from matplotlib import pyplot as plt
import numpy as np
import random
import copy
import pickle
import os


def findPlate(filepath="./_M2B3097.jpg"):

  #filepath = "./_M2B3091.jpg" #works well on this one
  #filepath = "./_M2B3097.jpg"
  #filepath = "./_DNF0596.jpg"
  #filepath = "./_DNF0618_cropped.jpg"
  #filepath = "./_DNF0618.jpg" #a hard one
  #filepath = "./_DNF0630.jpg"
  #filepath = "_N7B2040.jpg"
  #filepath = "_DNF0648.jpg"
  #filepath = "_DNF0395.jpg" - doesn't crop on numberplate
  #filepath= "_DNF0612.jpg"


  img = cv2.imread(filepath,0)
  ffimg = copy.deepcopy(img)
  #blur_img = cv2.blur(img,(10,10))
  blur_img = cv2.blur(img,(5,5))

  #sobel_img = cv2.Sobel(blur_img,cv2.CV_64F,1,0,3,1,0)
  sobel_img = cv2.Sobel(blur_img,-1,1,0)

  ret,threshold_img = cv2.threshold(sobel_img,0,255,cv2.THRESH_OTSU)
  #ret,threshold_img = cv2.threshold(sobel_img,0,255,cv2.THRESH_TOZERO)

  element = cv2.getStructuringElement(cv2.MORPH_RECT,(17,2))
  morphed_img = cv2.morphologyEx(threshold_img,cv2.MORPH_CLOSE,element)

  kernel = np.ones((2,7),np.uint8)
  morphed_img = cv2.erode(morphed_img,kernel,iterations = 6)
  morphed_img = cv2.dilate(morphed_img,kernel,iterations = 9)


  #plt.imshow(morphed_img,'gray')
  #plt.show()
  #kernel = np.ones((8,8),np.uint8)
  #morphed_img2 = morphed_img.copy()
  #morphed_img2 = cv2.dilate(morphed_img,kernel)
  #morphed_img2 = cv2.morphologyEx(threshold_img,cv2.MORPH_CLOSE,kernel)
  morphed_img, contours, hierarchy = cv2.findContours(morphed_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  #morphed_img, contours, hierarchy = cv2.findContours(morphed_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  #plt.imshow(morphed_img,'gray')
  #plt.show()


  def verifySizes(mr):
    error = 0.4
    aspect = 4.7272
    minArea = 20*aspect*20
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

  # print "before ", len(contours)

  rects = []
  newcontours = []

  for contour in contours:

    mr = cv2.minAreaRect(contour)
    if (verifySizes(mr)):
    #if (verifySizes(mr)) and cv2.contourArea(contour)/len(contour)<100:
      #print "contour area", cv2.contourArea(contour)
      rects.append(mr)
      newcontours.append(contour)

  # print "after ", len(rects)
  # print len(newcontours)

  # print morphed_img.shape
  plateList =  []

  for numContour,rect in enumerate(rects):
    box = cv2.boxPoints(rect)

    x = rect[0][0]
    y = rect[0][1]
    width = float(rect[1][0])
    height = float(rect[1][1])
    centerx = int(x)
    centery = int(y)

    result = cv2.circle(morphed_img,(centerx,centery), 3, (0,255,0), -1)
    minSize = min(width/2, height/2)

    box = np.int0(box)
    mask = np.zeros((morphed_img.shape[0]+2, morphed_img.shape[1] +2),np.uint8)
    mask[:] = 0
    #cv2.fillConvexPoly(mask,box.astype(np.int32),(255))

    #masksm= np.zeros((morphed_img.shape[0], morphed_img.shape[1] ),np.uint8)
    #masksm[:] = 0
    #cv2.fillConvexPoly(masksm,box,(255))

    #maskedimg = img & masksm

    #cv2.drawContours(mask,newcontours[numContour],-1,255,-1)
    # print mask.shape
    # print type(mask)
    #mask = cv2.drawContours(mask,newcontours[0],0,255,-1)
    # print centerx
    lo,hi = 30,30
    connectivity = 4
    flags = connectivity
    flags |= (255 << 8)
    flags |= cv2.FLOODFILL_FIXED_RANGE
    flags |= cv2.FLOODFILL_MASK_ONLY

    #flags |= cv2.FLOODFILL_MASK_ONLY
    numseeds  = 5
    seeds = np.zeros((numseeds,2),np.int16)
    for i in range(numseeds):
      # print random.randrange(int(centerx-minSize/2),int(centerx+minSize/2))
      seeds[i][0] = random.randrange(int(centerx-minSize/2),int(centerx+minSize/2))
      seeds[i][1] = random.randrange(int(centery-minSize/2),int(centery+minSize/2))
      #seeds[i]=[int(val) for val in seeds[i]]
      # print seeds[i]

      #cv2.floodFill(img,mask,tuple(seeds[i]),(255,255,255),lo,hi,flags)
      cv2.floodFill(img,mask,tuple(seeds[i]),(255,255,255),lo,hi,flags)

    floodPoints = np.where(mask == 255)
    floodPoints = [floodPoints[1],floodPoints[0]]
    floodPoints = np.transpose(floodPoints)

    newMR = cv2.minAreaRect(floodPoints)

    newBox = np.int0(cv2.boxPoints(newMR))
    #plt.imshow(mask,'gray')
    #plt.show()
    if (not verifySizes(newMR)):
       continue
    else:
      pass



  #
  #  # find the enclosing non rotated rectangle
  #  def enclosingRect(box):
  #    xMin = min([box[i][0] for i in range(3)])
  #    xMax = max([box[i][0] for i in range(3)])
  #    yMin = min([box[i][1] for i in range(3)])
  #    yMax = max([box[i][1] for i in range(3)])
  #    return (xMin,xMax,yMin,yMax)
  #
  #
  #  #
  #  #
  #
  #  xMin,xMax,yMin,yMax = enclosingRect(newBox)
  #
  #  center = ((xMin+xMax)/2,(yMin+yMax/2))
  #
    center = tuple(np.int0(newMR[0]))
    angle = newMR[-1]
    width = newMR[1][0]
    height = newMR[1][1]

    newAspectRatio = width/height

    if (newAspectRatio < 1):
      angle = angle+90
      width,height = height,width
    rotationMatrix = cv2.getRotationMatrix2D(center,angle,1)

    rotatedImg = cv2.warpAffine(img,rotationMatrix,(img.shape[1],img.shape[0]))
    croppedImg = cv2.getRectSubPix(rotatedImg, (int(width),int(height)), center)

    resultResized =  cv2.resize(croppedImg,(144,33))
    blurResult = cv2.blur(resultResized,(3,3))
    equalResult = cv2.equalizeHist(blurResult)
    plateList.append(equalResult)




  #  plt.imshow(croppedImg)
    #plt.imshow(rotatedImg)
  #  plt.show()
  #  plt.imshow(resultResized)
  #  plt.show()


    #
    #  # calculate a new rectangle for just the flood filled points
    #  pointsofinterest = []
    #  for x in range(xMin, xMax):
    #    for y in range(yMin, yMax):
    #      # we flip x y here because .shape returns y first.. confusing eh?
    #      try:
    #        if maskedimg[y][x] == 255:
    #          pointsofinterest.append([x,y])
    #      except:
    #        print "box is :", box
    #        print "shape is:", maskedimg.shape
    #        print "x,y:", x, y
    #        break
    #
    #  #print "there are: ", len(pointsofinterest), " ponits of interest"
    #  newMR = cv2.minAreaRect(np.array(pointsofinterest))
    #
    #  x = newMR[0][0]
    #  y = newMR[0][1]
    #  width = float(newMR[1][0])
    #  height = float(newMR[1][1])
    #  centerx = int(x)
    #  centery = int(y)
    #
    #  newBox = np.int0(cv2.boxPoints(newMR))
    #  #print newBox
    #

    #newimg = cv2.rectangle(img,(int(centerx - width/2),int(centery - height/2)),(int(centerx + width/2),int(centery+ height/2)),(128,128,128))
    #newimg = cv2.drawContours(img,[newBox],0,(128,128,128), 5)


    #new_img = cv2.circle(img,tuple(seeds[0]),100, (255, 255, 255), 10)
    #new_img = cv2.circle(img,(centerx,centery),10, (0, 0,0 ), 10)
    # print "this is rects"
    # print rects[0]
    # print "this is box"
    # print box
    # print int(rects[0][0][0]+rects[0][1][0])
  # ((10135, 912.6630249023438), (235.74269104003906, 64.00566864013672), -1.3639276027679443)


    # print seeds[0]

    #cv2.imshow('circle', img)
  #cv2.drawContours(img,contours,-1,(0,255,0),3)
  #if(len(newimg)):
  #  cv2.imshow('circle', newimg)
  #else:
  #  cv2.imshow('circle',img)

  #plt.subplot(2,1,1)
  #plt.imshow(morphed_img2)
  #plt.subplot(2,1,2),
  #plt.imshow(morphed_img)
  #plt.show()
  #cv2.imshow('image',sobel_img)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()

  try:
    inputFile = open("data.list2","r+b")
    inputList = pickle.load(inputFile)
    inputList = [i for i in inputList[0]]
    inputFile.close()
    outputFile = open("data.list2", "wb")

  except:
    outputFile = open("data.list2","wb")
    inputList = []


  print "before", len(inputList)
  for plate in plateList:
    inputList.append(plate)
  print len(inputList)
  pickle.dump([inputList],outputFile)
  outputFile.close()

def compileData():
  folderPath = "/Users/andy/work/pos3"
  for root, dirs, filenames in os.walk(folderPath):
    for f in filenames:
      if f[-3:] == "jpg":
        print f
        findPlate(folderPath + "/" + f)
