import sys
sys.path.append('/usr/local/opencv2/lib/python2.7/site-packages')
import cv2
import numberplate
from matplotlib import pyplot as plt
import pickle
import numpy as np
import copy

def descriptors(img):
  return img

def trainer(fileName="data.list",resultsFileName="results.txt"):
  svm_parms = dict (kernel_type=0L,svm_type=100L,C=2.67,gamma=5.383 )
  affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

  SZ=20
  bin_n = 16


  f = open(fileName, "r+b")
  p = pickle.load(f)
  f.close()
  trainData = np.float32(p[0]).reshape(-1,144*33)
  resultsFile = open(resultsFileName, "r")
  results = resultsFile.read()
  resultsFile.close()
  results = results.splitlines()
  results = np.float32([ int(i) for i in results])
  svm = cv2.SVM()
  svm.train(trainData,results, params=svm_parms)
  #svm.save('svm_data.dat')
  return svm

def tester(fileName="data.list.iterations4"):
  testFile = open("data.list.iterations4", "r+b")
  p = pickle.load(testFile)
  testFile.close()
  testData = np.float32(p[0]).reshape(-1,144*33)
  result = svm.predict_all(testData)
  print result

def verifyCharSize(rect,img):
  center = rect[0]
  width = rect[1][0]
  height = rect[1][1]
  if height == 0:
    return False
  minHeight = 15
  maxHeight = 28
  aspect = 45.0/77
  charAspect = float(width/height)
  minAspect = 0.2
  error = 0.35
  maxAspect = aspect+aspect*error
  croppedImg = cv2.getRectSubPix(img, (int(width),int(height)), center)
  print croppedImg.shape

  numWhite  = np.where(croppedImg == 255)

  print numWhite
  #totalPixels = c



def readPlate(img):
  ret,threshold_img = cv2.threshold(img,60,255,cv2.THRESH_BINARY_INV)
  cont_img = copy.deepcopy(threshold_img)
  contours, hierarchy = cv2.findContours(cont_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  #cv2.drawContours(img,contours,-1,(0,255,0),1)
  #print len(contours)
  for contour in contours:
    mr = cv2.minAreaRect(contour)
    verifyCharSize(mr,threshold_img)

  #  cv2.drawContours(img,contour,-1,(0,255,0),1)
  #  plt.imshow(img,'gray')
  #  plt.show()

def openImage(fileName="data.list"):
   f = open(fileName, "r+b")
   p = pickle.load(f)
   f.close()
   img = p[0][1]
   readPlate(img)


openImage()
#
#for img in p[0]:
#  plt.imshow(img)
#  plt.show()



