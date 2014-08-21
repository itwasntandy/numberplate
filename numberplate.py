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
import platereader


def findPlate(filepath="./images/_M2B3097.jpg"):

  #filepath = "./images/_M2B3091.jpg" #works well on this one
  #filepath = "./images/_M2B3097.jpg"
  #filepath = "./images/_M2B3102.jpg"
  #filepath = "./images/_M2B3099.jpg"
  #filepath = "./images/_DNF0596.jpg"
  #filepath = "./images/_DNF0618_cropped.jpg"
  #filepath = "./images/_DNF0618.jpg" #a hard one gets H53YAT, should be H643 YAT
  #filepath = "./images/_DNF0630.jpg" # floodfill messes this one up
  #filepath = "./images/_DNF0634.jpg" # gets YG2UTJ should be YG12 UTJ.
  #filepath = "./images/_N7B2040.jpg"
  #filepath = "./images/_DNF0648.jpg" # finds AEUU5 YUS - should be AE05 YUS
  #filepath = "./images/_DNF0395.jpg" #- doesn't crop on numberplate
  #filepath= "./images/_DNF0612.jpg"
  #filepath= "./images/_DNF0663.jpg"


  img = cv2.imread(filepath,0)
  ffimg = copy.deepcopy(img)
  #blur_img = cv2.blur(img,(10,10))
  blur_img = cv2.blur(img,(5,5))

  #sobel_img = cv2.Sobel(blur_img,cv2.CV_64F,1,0,3,1,0)
  sobel_img = cv2.Sobel(blur_img,-1,1,0)
  #plt.imshow(sobel_img, 'gray')
  #plt.show()
  ret,threshold_img = cv2.threshold(sobel_img,0,255,cv2.THRESH_OTSU)
  #ret,threshold_img = cv2.threshold(sobel_img,0,255,cv2.THRESH_TOZERO)

  element = cv2.getStructuringElement(cv2.MORPH_RECT,(28,5))
  morphed_img = cv2.morphologyEx(threshold_img,cv2.MORPH_CLOSE,element)

  kernel = np.ones((2,7),np.uint8)
  morphed_img = cv2.erode(morphed_img,kernel,iterations = 6)
  morphed_img = cv2.dilate(morphed_img,kernel,iterations = 9)
  #plt.imshow(morphed_img, 'gray')
  #plt.show()



  #kernel = np.ones((8,8),np.uint8)
  morphed_img, contours, hierarchy = cv2.findContours(morphed_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  #new_img = cv2.drawContours(threshold_img, contours,-1,128,-1)
  #plt.imshow(new_img,'gray')
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
    #if (verifySizes(mr)):
    if True:
    #if (verifySizes(mr)) and cv2.contourArea(contour)/len(contour)<100:
      #print "contour area", cv2.contourArea(contour)
      rects.append(mr)
      newcontours.append(contour)

  # print "after ", len(rects)
  #print len(newcontours)
  new_img = cv2.drawContours(threshold_img, newcontours,-1,128,-1)
  #plt.imshow(new_img,'gray')
  #plt.show()

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
    blurResult = cv2.blur(resultResized,(1,1))
    equalResult = cv2.equalizeHist(blurResult)
    plateList.append(equalResult)

  numberlist = []
  numbnospaces = []
  for plateCandidate in plateList:
    #plt.imshow(plateCandidate,'gray')
    #plt.show()
    answer = platereader.readPlate(plateCandidate)
    if answer:
        numberlist.append(answer)
        nospace = answer.replace(" ", "")
        numbnospaces.append(nospace)

  if len(numberlist) == 0:
      return
  longest = np.argmax([len(num) for num in numbnospaces])
  print numberlist[longest]
#  try:
#    inputFile = open("data.list2","r+b")
#    inputList = pickle.load(inputFile)
#    inputList = [i for i in inputList[0]]
#    inputFile.close()
#    outputFile = open("data.list2", "wb")
#
#  except:
#    outputFile = open("data.list2","wb")
#    inputList = []
#
#
#  print "before", len(inputList)
#  for plate in plateList:
#    inputList.append(plate)
#  print len(inputList)
#  pickle.dump([inputList],outputFile)
#  outputFile.close()
#
#def compileData():
#  folderPath = "/Users/andy/work/nplates"
#  for root, dirs, filenames in os.walk(folderPath):
#    for f in filenames:
#      if f[-3:] == "jpg":
#        print f
#        findPlate(folderPath + "/" + f)

#compileData()

#if len(sys.argv) > 1:
    #typically it will be run with python numberplate.py filename.jpg
    #arg argv[1] (2nd arg) is actually the numberplate
#    findPlate(sys.argv[1])
#elif len(sys.argv) == 1 and sys.argv[0] != "numberplate.py":
    #deal with the case where numberplate.py is executable,
    #and there is still an argument passed
#    findPlate(sys.argv[0])
#else:
#    findPlate()



