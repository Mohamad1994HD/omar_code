import argparse
import math
import cv2
import numpy as np

import imutils

from shape import *


class Detector(object):
    GKERNEL = (5,5)
    DEFAULT_ANCHOR = Shape.CIRCLE
    DEFAULT_BIN = Shape.SQUARE

    def __init__(self, img=None):
        assert img is not None
        self.image = img
        #
        self.thres = None
        #
        self.objects = ShapeList()
        self.bin_template = None
        #
        self.preprocess()
        self.digest_contours()

    def highlight(self):
        self.objects.highlight(self.image)

    def preprocess(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.GKERNEL, 0)
        self.thres = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]

    def digest_contours(self, compensate=False):
        cnts = cv2.findContours(self.thres.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        for c in cnts:
            self.objects.append(ShapeDescriptor(c))
        
        if not compensate:
            if self.objects.min.is_type(self.DEFAULT_BIN):
                self.bin_template = self.objects.min

        tocompensate = False
        # Check if intersection, to compensate
        for item in self.objects:
            if (not item.is_type(self.DEFAULT_ANCHOR)) \
                    and (item.size > 1.5*self.bin_template.size):
                tocompensate = True
                x,y,w,h = item.bound_rect    
                wr, hr = self.bin_template.bound_rect[2:]
                ratiow = int(w/float(wr))
                ratioh = int(h/float(hr))
                if h/float(w) > 1:
                    for i in xrange(ratioh):
                        cv2.line(self.thres, (x, y+hr),(x+w, y+hr), 0, 1)
                else:
                    for i in xrange(ratiow):
                        cv2.line(self.thres, (x+wr, y),(x+wr, y+h), 0, 1)
        if tocompensate:
            del self.objects[:]
            self.digest_contours(compensate=True)
            return
        #print [x.type for x in self.objects]
#        x = list(set(self.objects).difference(self.objects.query(self.DEFAULT_ANCHOR)+self.objects.query(self.DEFAULT_BIN)))[0]
#        print x.bound_rect
#        print len(self.objects.query(self.DEFAULT_BIN)) 
#        if tocompensate:
#            # Intersection exist, run a convolution kernel
#            # Remove the anchors
#            for item in self.objects:
#                if item.is_type(self.DEFAULT_ANCHOR):
#                    # needs refactor
#                    cv2.circle(self.thres, item.position, int(math.sqrt(item.size/math.pi))+2, (0, 0, 0), -1)
#            # Slide the kernel, and count
#            x = int(math.sqrt(mini.size))
#            kernel = np.full((x,x), 255, dtype='int')
#            (ih,iw) = self.thres.shape[:2]
#            (kh,kw) = kernel.shape[:2]
#            pad = (kw - 1) / 2
#            upperStart = sorted(self.objects.query(self.DEFAULT_BIN), key=lambda x:x.position[1])[0].position[1] - mini.bound_rect[3]/2 
#            leftStart = sorted(self.objects.query(self.DEFAULT_BIN), key=lambda x:x.position[0])[0].position[0] - mini.bound_rect[2]/2
#            print leftStart
#            for y in np.arange(upperStart, ih, 14):
#                cv2.line(self.image, (0, y), (iw, y), 0, 1)
#            for x in np.arange(leftStart, iw, 2*pad+1):
#                cv2.line(self.image, (x, 0), (x, ih), 0, 1)
#
#            self.thres = cv2.copyMakeBorder(self.thres, pad, pad, pad, pad,
#                            cv2.BORDER_CONSTANT)
#            for y in np.arange(pad, ih + pad, 2*pad+1):
#                for x in np.arange(pad, iw + pad, 2*pad+1):
##                   roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
#                    roi = self.thres[y - pad:y + pad + 1, x - pad:x + pad + 1]
#                    k = np.logical_and(roi, kernel).sum()
#                    cv2.circle(self.thres, (x,y), 1, 100, -1)
#                    if k > 0.5 * roi.size:
#                        cv2.circle(self.thres, (x,y), 3, 0, -1)
#                        #self.thres[y-pad:y+pad+1, x-pad:x+pad+1] = kernel
            #print self.objects.query(self.DEFAULT_BIN).min.size

    def validate(self):
        pass
    
    @property
    def num_of_objects(self):
        return len(self.objects)
    @property
    def num_of_anchors(self):
        return self.objects.count(self.DEFAULT_ANCHOR)
    @property
    def num_of_bins(self):
        return self.objects.count(self.DEFAULT_BIN) 

def load_img(path=None):
    assert path is not None
    #
#    h, w = img.shape[:2]
#    if h>500 and w>500:
#        img = cv2.resize(img, (600, 600)) 
    return cv2.imread(path)


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
    ap.add_argument("-o", "--output", required=False, 
            help="path to the output image")
            
    args = vars(ap.parse_args())
     
    # load the image, convert it to grayscale, blur it slightly,
    # and threshold it
    image = cv2.imread(args["image"])
    cv2.imshow("original", image)
    detector = Detector(image)
    detector.highlight()
    cv2.imshow("thres", detector.thres)
    #
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]
#    # find contours in the thresholded image
#    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#                cv2.CHAIN_APPROX_SIMPLE)
#    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
#    for c in cnts:
#            # compute the center of the contour
#        try:
#            M = cv2.moments(c)
#            cX = int(M["m10"] / M["m00"])
#            cY = int(M["m01"] / M["m00"])
#            # draw the contour and center of the shape on the image
#            sh = estimate_shape(c)
#            color = (0, 255, 0)
#            if sh == Shape.CIRCLE:
#                color = (255, 0, 0)
#            elif sh == Shape.RECTANGLE:
#                color = (0, 0, 255)
#            cv2.drawContours(image, [c], -1, color, 2)
#            cv2.circle(image, (cX, cY), 1, (255, 255, 255), -1)
#        except:
#            pass
    cv2.imshow("final", image)
    if args["output"]:
        cv2.imwrite(args["output"], image)
    cv2.waitKey(0)
