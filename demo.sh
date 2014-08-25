#!/bin/bash

echo "starting Demo"

echo ""
open ./images/_M2B3099.jpg
echo "running python read-single.py images/_M2B3099.jpg..."
echo ""
python read-single.py images/_M2B3099.jpg

echo ""
read -p "next..."
echo ""
open images/_M2B3097.jpg
echo ""
echo "running python read-single.py images/_M2B3097.jpg..."
echo ""
python read-single.py images/_M2B3097.jpg 
echo ""
read -p "next..."
open images/_M2B3115.jpg
echo ""
echo "running python read-single.py images/_M2B3115.jpg..."
echo ""
python read-single.py images/_M2B3115.jpg
echo ""
read -p "next..."
open images/_M2B3096.jpg
echo ""
echo "running python read-single.py images/_M2B3096.jpg..."
echo ""
python read-single.py images/_M2B3096.jpg
echo ""
#read -p "next..."
#open images/_M2D8255.jpg
#echo ""
#echo "running read-single finder"
##echo "..."
#python read-single.py images/_M2D8255.jpg
echo ""
read -p "next..."
open images/_M2D8388.jpg
echo ""
echo "running python read-single.py images/_M2D8388.jpg..."
echo ""
python read-single.py images/_M2D8388.jpg
echo ""


