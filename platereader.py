import sys
sys.path.append('/usr/local/opencv2/lib/python2.7/site-packages')
import cv2
#import numberplate
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.cm as cm
import pickle
import numpy as np
import copy
import pytesseract

def tester(fileName="data.list.iterations4"): #,outputFileName="default-plate-image.jpg"):
  svm = cv2.SVM()
  svm.load('svm_data.dat')
  testFile = open("data.list.iterations4", "r+b")
  p = pickle.load(testFile)
  testFile.close()

  testData = np.float32(p[0][3]).reshape(-1,144*33)
  result = svm.predict(testData)
  if result:
      testDataAspect = np.float32(p[0][3])
      return testDataAspect

def deskew(img):
    SZ=img.shape[0]
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,img.shape,flags=affine_flags)
    return img

def readLetter(contour,img,refine=False):
  x,y,w,h = cv2.boundingRect(contour)
  if h == 0:
    return False
  minHeight = 15
  maxHeight = 28
  aspect = 45.0/77
  charAspect = float(w)/h
  minAspect = 0.2
  error = 0.35
  maxAspect = aspect+aspect*error
  croppedImg = cv2.getRectSubPix(img, (int(w),int(h)), (int(x+w/2),int(y+h/2)))

  numWhite = len(np.transpose(np.nonzero(croppedImg)))
  percentPixels = float(numWhite)/(w*h)
  #if percentPixels > 0.2 and charAspect > minAspect and charAspect < maxAspect and h < maxHeight and h > minHeight:
  if h < maxHeight and h > minHeight:
      vertImg= cv2.copyMakeBorder(croppedImg,5,5,15,15,cv2.BORDER_CONSTANT,(0,0,0))
      ret,vertImg = cv2.threshold(vertImg,2,255,cv2.THRESH_BINARY_INV)
      kernel = np.ones((1,1),np.uint8)
      vertImg = cv2.erode(vertImg,kernel,iterations =2)
      if refine:
        kernel = np.ones((2,2),np.uint8)
        vertImg = cv2.erode(vertImg,kernel,iterations = 3)
        #vertImg = cv2.dilate(vertImg,kernel,iterations = 2)

      j = Image.fromarray(vertImg)
      #plt.matshow(vertImg, cmap = cm.Greys_r)
      #plt.show()

      letter =  pytesseract.image_to_string(j,config='./tesseract-config')
      #print "letter is ", letter
      if len(letter) >0:
          #print letter
          return letter
      elif len(letter) == 0 and refine== True:
        #still false
        return " "
      return "_"
  return  False


def readPlate(img):
  origimg = copy.deepcopy(img)
  ret,threshold_img = cv2.threshold(img,60,255,cv2.THRESH_BINARY_INV)


  cont_img = copy.deepcopy(threshold_img)
  cont_img, contours, hierarchy = cv2.findContours(cont_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
  letters = []
  for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)

    if len(contour) < 4:
        continue
    letter = readLetter(contour,threshold_img)
    if letter == "_" or letter == "U":
        letter = readLetter(contour,threshold_img,True)

    if letter == "I" or letter == "l":
        letter = "1"
        letters.append((letter,x))
    elif not letter == False:
        letters.append((letter,x))

 # trueletters = [letters[i][0] for i in  len(letters) if not letters[i][0] == " "]


  letters = sorted(letters, key=lambda y: y[1])
  numberplate = [letters[i][0] for i in range(len(letters))]
  numberplate = ''.join(numberplate)
  #cv2.drawContours(threshold_img,contours,-1,(128,128,128),1)
  ## plt.imshow(origimg,'gray')
  #plt.matshow(threshold_img, cmap = cm.Greys_r)
  #plt.show()

  #if len(numberplate) >3 and len(numberplate) < 10:
  #    return numberplate
  #else:
  #    return
  return numberplate

def openImage(fileName="data.list.iterations4"):

   #f = open(fileName, "r+b")
   #f = open('./numberplate-german.jpg', "r+b")
   img = cv2.imread("./numberplate-german.jpg", 0)
   #p = pickle.load(f)
   #f.close()
   #iimg = p[0][4] # this p[0][4] was good
   #img = p[0][7] # this p[0][7] was much harder somewhat blurry.
   print readPlate(img)

#openImage()

