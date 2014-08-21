#!/bin/bash

echo "starting Demo"

echo "open _M2B3099.jpg"
open ./images/_M2B3099.jpg

echo "running read-single finder"
echo "..."
python read-single.py images/_M2B3099.jpg
echo ""
read -p "press enter to continue"
echo ""
open images/_M2B3097.jpg

echo ""

echo "running read-single finder"
echo "..."
python read-single.py images/_M2B3097.jpg 

read -p "press enter"
open images/_M2B3115.jpg
echo ""
echo "running read-single finder"
echo "..."
python read-single.py images/_M2B3115.jpg

read -p "press enter"
open images/_M2B3096.jpg
echo ""
echo "running read-single finder"
echo "..."
python read-single.py images/_M2B3096.jpg

#read -p "press enter"
#open images/_M2D8255.jpg
#echo ""
#echo "running read-single finder"
##echo "..."
#python read-single.py images/_M2D8255.jpg

read -p "press enter"
open images/_M2D8388.jpg
echo ""
echo "running read-single finder"
echo "..."
python read-single.py images/_M2D8388.jpg


