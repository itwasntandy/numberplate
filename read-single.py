#!/usr/bin/python
import sys
import numberplate

if len(sys.argv) > 1:
    #typically it will be run with python numberplate.py filename.jpg
    #arg argv[1] (2nd arg) is actually the image of the car
    numberplate.findPlate(sys.argv[1])
elif len(sys.argv) == 1 and sys.argv[0] != "numberplate.py":
    #deal with the case where numberplate.py is executable,
    #and there is still an argument passed
    numberplate.findPlate(sys.argv[0])
else:
    numberplate.findPlate()



