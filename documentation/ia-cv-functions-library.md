# Functions for Interactive Architecture CV-part

## Utilities

- `midpoint` finding midpoint between two (2D) points 
- `pix2metric` determine *real* distance per pixel
- `getbbox` determine the bounding box for a given contour
- `getcm` determine the center of mass given a rectangular bounding box

## Image processing

- _simple_
  - `getshape` prints/gets the shape of image 
  - `getsize` prints/gets the size (in pixels and optionally in pre-defined metric) of an image
  - `imresize` resizes the image 
  - `rgb2gray` from color to grayscale 
  - `2blackborder` makes bordering pixel (upto some offset) black
- _filter_
  - `edge` finds the edges in an image (grayscale)
  - `dilate` dilates an image (grayscale)
  - `erode` erodes an image (grayscale)
  - `blur` blurs an image (blur)
- _advanced_
  - `findcnt` finds closed contours in an image (grayscale)
  - `warp` warps an image given a rectangular frame

## Image plotting

- `imshow` plots an image inline
- `drawbbox` draws the bounding box (on top of original image)
- `drawcm` draws the center of mass (on top of original image)
