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

#PACKAGES
from scipy.spatial import distance as dist
import cv2
from imutils import perspective
from imutils import contours
import imutils
import matplotlib.pyplot as plt
import numpy as np

#SETTINGS
color_edge    = (0, 255, 0)   #bounding box
color_text    = (0, 255, 0)   #text
color_line    = (255, 0, 255) #middle lines (crossing in middle of object)
color_bpoints = (0, 0, 255)   #bounding box angle points
color_mpoints = (255, 0, 0)   #bounding box middle points
color_cross   = (0, 0, 255)   #center of mass color


#UTILITIES
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def pixpermetric(image, ref, metric="cm", show=False):
    h, w = img.shape[0], img.shape[1]
    if ref.size()==1: 
        pixpm = w/ref
        if show: print(f"Pixels per {metric} in X and Y = {pixpm:.2f}")
    elif ref.size()==2: 
        pixpm = [h/refh, w/refw]
        refw, refh = ref[0], ref[1]
        if show: print(f"Pixels per {metric} in Y = {pixpm[0]:.2f} \nPixels per {metric} in X = {pixpm[1]:.2f}")
    return pixpm

def pix2metric(dim, pixpm, metric="cm",show=False):
    dimY, dimX = dim[0], dim[1]
    if pixpm.size() == 2: d = [dimY/pixpm[0], dimX/pixpm[1]]
    elif pixpm.size() == 1: d = [dimY/pixpm, dimX/pixpm]
    if show: print(f"The size of the bounding box frame is: {d[0]:.2f}{metric} x {d[1]:.2f}{metric} (height x width) /n")
    return d

def getbbox(cnts):
    bboxs = [] #create empty list to store the bounding box data
    for c in cnts:
        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        
        # order the points in the contour such that 
        # they would appear in top-left, 
        # top-right, bottom-right, and bottom-left order
        bboxs.append(perspective.order_points(box))
    
    return bboxs

def getcm(bboxs):
    cms = [] #create empty list to store the center of mass data
    for box in bboxs:
        cm_y = np.sum(box[:,0])/4
        cm_x = np.sum(box[:,1])/4
        cms.append([cm_y, cm_x])
    return cms

def draw_cross(img, center, color, d=10, t=3):
    cv2.line(img,
             (center[0] - d, center[1] - d), (center[0] + d, center[1] + d),
             color, t, cv2.LINE_AA, 0)
    cv2.line(img,
             (center[0] + d, center[1] - d), (center[0] - d, center[1] + d),
             color, t, cv2.LINE_AA, 0) 

#IMAGE PROCESSING

#_simple
def getshape(img):
    h, w, dim = img.shape
    print(f"Shape of the image is {h} x {w} x {dim} (height x width x dimension)")
    
def imresize(img, w_resize=500):
    h, w = img.shape[0], img.shape[1]
    h_resize = int(h / w * w_resize)
    dim = (w_resize, h_resize)
    img_resize = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img_resize

def rgb2gray(img):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def border2black(img, off=5):
    img[:,  :off] = 0
    img[:, -off:] = 0
    img[:off,  :] = 0
    img[-off:, :] = 0
    return img


#_filter
def edge(img, e_range=(75,100)):
    return cv2.Canny(img, e_range[0], e_range[1]) #simple edge detector

def dilate(img, ksize=3, iterations=1):
    return cv2.dilate(img, (ksize, ksize), iterations=iterations)

def erode(img, ksize=3, iterations=1):
    return cv2.erode(img, (ksize, ksize), iterations=iterations)

def blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


# _advanced
def findcnts(img, a_min=100):
    
    # `cnts_` is a list with all contours, `cnts` with the real ones
    cnts_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_ = imutils.grab_contours(cnts_)
    (cnts_, _) = contours.sort_contours(cnts_)
    
    cnts = [] #create empty list to store the 'real' contour data
    
    for c in cnts_:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < a_min: continue
        cnts.append(c)

    return cnts

def warp(img, cnt):

    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect)
    box  = np.int0(box)
    w_rect = int(rect[1][0])
    h_rect = int(rect[1][1])
    
    #get source and distance points
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, h_rect-1],
                        [0, 0],
                        [w_rect-1, 0],
                        [w_rect-1, h_rect-1]], dtype="float32")
    
    #perspective transform and warping of the image based on rectangular reference frame
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w_rect, h_rect))
    
    return warped


#IMAGE PLOTTING

def imshow(img, figsize=[10, 10], axis=False):
    imgp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #color annotation is different for plt.imshow and cv2.imread
    plt.figure(figsize=figsize)
    if axis==False: plt.axis('off')    
    plt.imshow(imgp)
    
def drawbbox(img, bboxs, pixpm):
    
    #copy image
    img = img.copy()
    
    for bbox in bboxs:
        
        #draw contours
        cv2.drawContours(img, [bbox.astype("int")], -1, color_edge, 2)
        
        # loop over the original points and draw them
        for (x, y) in bbox:
            cv2.circle(img, (int(x), int(y)), 5, color_bpoints, -1)
        
        # unpack the ordered bounding box, then compute the midpoint 
        # between the top-left and top-right coordinates, followed 
        # by the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = bbox
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points, 
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(img, (int(tltrX), int(tltrY)), 5, color_mpoints, -1)
        cv2.circle(img, (int(blbrX), int(blbrY)), 5, color_mpoints, -1)
        cv2.circle(img, (int(tlblX), int(tlblY)), 5, color_mpoints, -1)
        cv2.circle(img, (int(trbrX), int(trbrY)), 5, color_mpoints, -1)

        # draw lines between the midpoints
        cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
            color_line, 2)
        cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
            color_line, 2)

        # compute the Euclidean distance between the midpoints
        dimA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dimB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then compute 
        # it as the ratio of pixels to supplied metric (in this case, inches)
        d = pix2metric([dimA, dimB], pixpm)

        # draw the object sizes on the image
        cv2.putText(img, "{:.2f} cm".format(d[0]),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color_text, 2)
        cv2.putText(img, "{:.2f} cm".format(d[1]),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color_text, 2)
    
    imshow(img)
        
def drawcm(img, cms, pixpm):
    
    for cm in cms:

        # distance per metric
        cm_metric = pix2metric(cm, pixpm)

        # draw cross in the middle
        draw_cross(img, (int(cm[0]), int(cm[1])), color_cross)

        # draw the object sizes on the image
        cv2.putText(img, "{:.2f} cm".format(cm_metric[0]),
            (int(cm[0] - 25), int(cm[1] - 25)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color_text, 2)
        cv2.putText(img, "{:.2f} cm".format(cm_metric[1]),
            (int(cm[0] + 25), int(cm[1] + 6)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color_text, 2)
    
    #show image
    imshow(img)
