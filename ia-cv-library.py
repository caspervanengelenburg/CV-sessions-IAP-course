'''
Package for 1:1 Interactive Architecture Prototypes course - Computer vision part

Consist of the following functions: 

UTILITIES

  `midpoint` finding midpoint between two (2D) points 
  `pix2metric` determine *real* distance per pixel
  `getbbox` determine the bounding box for a given contour
  `getcm` determine the center of mass given a rectangular bounding box

IMAGE PROCESSING

_simple
  `getshape` prints/gets the shape of image 
  `getsize` prints/gets the size (in pixels and optionally in pre-defined metric) of an image
  `imresize` resizes the image 
  `rgb2gray` from color to grayscale 
  `2blackborder` makes bordering pixel (upto some offset) black
_filter
  `edge` finds the edges in an image (grayscale)
  `dilate` dilates an image (grayscale)
  `erode` erodes an image (grayscale)
  `blur` blurs an image (blur)
_advanced
  `findcnt` finds closed contours in an image (grayscale)
  `warp` warps an image given a rectangular frame

IMAGE PLOTTING

  `imshow` plots an image inline
  `drawbbox` draws the bounding box (on top of original image)
  `drawcm` draws the center of mass (on top of original image)

'''
