import sys
sys.path.append('/usr/local/opencv2/lib/python2.7/site-packages')
import cv2
import numberplate
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pickle
import numpy as np
import copy

def tester(fileName="data.list.iterations4"): #,outputFileName="default-plate-image.jpg"):
  svm = cv2.SVM()
  svm.load('svm_data.dat')
  testFile = open("data.list.iterations4", "r+b")
  p = pickle.load(testFile)
  testFile.close()

  #testData = np.float32(p[0][0]).reshape(-1,144*33)
  testData = np.float32(p[0][3]).reshape(-1,144*33)
  #plt.matshow(p[0][3])
  #plt.show()
  result = svm.predict(testData)
  if result:
      testDataAspect = np.float32(p[0][3])
      #cv2.imwrite(outputFileName,testDataLong)
      return testDataAspect

def verifyCharSize(contour,img):
  x,y,w,h = cv2.boundingRect(contour)
  #rect = minAreaRect(contour)
  #box = cv2.cv.BoxPoints(rect)
  #minX = min([box[i][0] for i in range(3)])
  #minY = min([box[i][1] for i in range(3)])
  #maxX = max([box[i][0] for i in range(3)])
  #maxY = max([box[i][1] for i in range(3)])
  #center = ((minX+maxX)/2,(minY+maxY)/2)
  #width = maxX-minX
  #height = maxY-minY
  #width = width*1.1
  #height = height*1.1
  if h == 0:
    return False
  minHeight = 15
  maxHeight = 28
  aspect = 45.0/77
  charAspect = float(w)/h
  minAspect = 0.2
  error = 0.35
  maxAspect = aspect+aspect*error
  #croppedImg = cv2.getRectSubPix(img, (int(width),int(height)), (int(center[0]),int(center[1])))
  croppedImg = cv2.getRectSubPix(img, (int(w),int(h)), (int(x+w/2),int(y+h/2)))
  #ret,threshold_img = cv2.threshold(croppedImg,60,255,cv2.THRESH_BINARY)

  #numWhite  = np.where(croppedImg == 255)
  numWhite = len(np.transpose(np.nonzero(croppedImg)))
  #percentPixels = float(numWhite)/(width*height)
  percentPixels = float(numWhite)/(w*h)
  print "percent pixels: ", percentPixels
  #if percentPixels > 0.6 and charAspect > minAspect and charAspect < maxAspect and height < maxHeight and height > minHeight:
  if percentPixels > 0.2 and charAspect > minAspect and charAspect < maxAspect and h < maxHeight and h > minHeight:
      plt.matshow(croppedImg, cmap = cm.Greys_r)
      plt.show()
      return True
  print "this is a fake"
  print "char Aspect: ", charAspect, " height:width: ", h,"x",w
  plt.matshow(croppedImg, cmap = cm.Greys_r)
  plt.show()
  return False


  #print numWhite
  #totalPixels = c



def readPlate(img):
  origimg = copy.deepcopy(img)
  ret,threshold_img = cv2.threshold(img,60,255,cv2.THRESH_BINARY_INV)
  #kernel = np.ones((1,1),np.uint8)
  #threshold_img = cv2.erode(threshold_img,kernel,iterations = 2)
  #threshold_img = cv2.dilate(threshold_img,kernel,iterations = 2)
  cont_img = copy.deepcopy(threshold_img)
  contours, hierarchy = cv2.findContours(cont_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
  #cv2.drawContours(img,contours,-1,(0,255,0),1)
  for contour in contours:
    #mr = cv2.minAreaRect(contour)
    #x,y,w,h = cv2.boundingRect(contour)

    if len(contour) < 5:
        continue
    #box = cv2.cv.BoxPoints(mr)
  #  print "mr is "
  #  print mr
  #  print "end of mr"
  #  print "box is"
  #  print box
  #  print "this is end of that box"
    #cv2.rectangle(img,(int(box[0][0]),int(box[0][1])),(int(box[2][0]),int(box[2][1])),(128,128,128))
    if verifyCharSize(contour,threshold_img):
        print "woohoo"

  cv2.drawContours(threshold_img,contours,-1,(128,128,128),1)

  #plt.subplot(2,1,1), plt.imshow(img,'gray')
  #plt.subplot(2,1,2), plt.imshow(origimg,'gray')
  plt.matshow(threshold_img, cmap = cm.Greys_r)
  plt.show()
  #  img = copy.deepcopy(origimg)
  #  plt.imshow(img,'gray')
  #  plt.show()

def openImage(fileName="data.list.iterations4"):
   f = open(fileName, "r+b")
   p = pickle.load(f)
   f.close()
   img = p[0][6]
   readPlate(img)


#tester()
openImage()
#
#for img in p[0]:
#  plt.imshow(img)
#  plt.show()



