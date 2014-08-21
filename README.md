# Numberplate (license place) reader

With a goal to understand more of what happens underneath the hood of OpenCV, Nava and I set out to build a tool to identify likely locations for a license plate on a car, and attempt to read it using OCR.

## Usage

### To scan a single image

* python read-single.py filename.jpg

### To scan a directory of images

* python read-many.py /path/to/images

## TODO

* improve performance through optimizations
* improve accuracy through training sets, or tesseract changes
* look for other information on cars (e.g. sponsor logos)

