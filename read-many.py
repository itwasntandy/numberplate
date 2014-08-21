#!/usr/bin/python


import sys
sys.path.append('/usr/local/opencv2/lib/python2.7/site-packages')
import os
import numberplate

#This works by scanning a directory of images, and outputting the plate of each

def directoryScan(folderPath = "/Users/andy/work/nplates"):
  for root, dirs, filenames in os.walk(folderPath):
    for f in filenames:
      if f[-3:] == "jpg":
        print f
        numberplate.findPlate(folderPath + "/" + f)


if len(sys.argv) > 1:
    #typically it will be run with python numberplate.py /path/to/images
    #arg argv[1] (2nd arg) is actually the directory
    directoryScan(sys.argv[1])
elif len(sys.argv) == 1 and sys.argv[0] != "numberplate.py":
    #deal with the case where numberplate.py is executable,
    #and there is still an argument passed
    directoryScan(sys.argv[0])
else:
    #default to the original specified path
    directoryScan()
