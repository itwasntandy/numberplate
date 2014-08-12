import sys
sys.path.append('/usr/local/opencv2/lib/python2.7/site-packages')
import cv2
import numberplate
from matplotlib import pyplot as plt
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

def verifyCharSize(rect,img):
  center = rect[0]
  width = rect[1][1]
  height = rect[1][0]
  if height == 0:
    return False
  minHeight = 15
  maxHeight = 28
  aspect = 45.0/77
  charAspect = float(width/height)
  minAspect = 0.2
  error = 0.35
  maxAspect = aspect+aspect*error
  croppedImg = cv2.getRectSubPix(img, (int(width),int(height)), (int(center[1]),int(center[0])))
  #ret,threshold_img = cv2.threshold(croppedImg,60,255,cv2.THRESH_BINARY)

  #numWhite  = np.where(croppedImg == 255)
  numWhite = len(np.transpose(np.nonzero(croppedImg)))
  percentPixels = float(numWhite)/(width*height)
  if percentPixels > 0.6 and charAspect > minAspect and charAspect < maxAspect and height < maxHeight and height > minHeight:
      return True
  #print percentPixels
  #plt.imshow(croppedImg, 'gray')
  #plt.show()
  return False


  #print numWhite
  #totalPixels = c



def readPlate(img):
  origimg = copy.deepcopy(img)
  ret,threshold_img = cv2.threshold(img,60,255,cv2.THRESH_BINARY_INV)
  cont_img = copy.deepcopy(threshold_img)
  contours, hierarchy = cv2.findContours(cont_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  #cv2.drawContours(img,contours,-1,(0,255,0),1)
  for contour in contours:
    print len(contour)
    mr = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(mr)
    print box
    cv2.rectangle(img,(int(box[0][0]),int(box[0][1])),(int(box[1][0]),int(box[1][1])),(128,128,128))
    if verifyCharSize(mr,threshold_img):
        print "woohoo"

    cv2.drawContours(img,contour,-1,(127,127,127),1)

    plt.subplot(2,1,1), plt.imshow(img,'gray')
    plt.subplot(2,1,2), plt.imshow(origimg,'gray')
    plt.matshow(img)
    plt.show()
    img = copy.deepcopy(origimg)
  #  plt.imshow(img,'gray')
  #  plt.show()

def openImage(fileName="data.list.iterations4"):
   f = open(fileName, "r+b")
   p = pickle.load(f)
   f.close()
   img = p[0][4]
   readPlate(img)


#tester()
openImage()
#
#for img in p[0]:
#  plt.imshow(img)
#  plt.show()



