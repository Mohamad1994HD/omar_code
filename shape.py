import math
from enum import Enum
import cv2


class Shape(Enum):
    UNDEFINED = 0
    CIRCLE = 1 
    SQUARE = 2 
    POLYGON = 3 
    TRIANGLE = 4
    RECTANGLE = 5

class Color(object):
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)

class ShapeDescriptor(object):
    SQUARE_RATIO_LIMITS = (0.70, 1.5)

    def __init__(self, contour):
        self.contour = contour
        #
        self.moments_ = cv2.moments(contour)
        #
        self.area = 0.0 
        self.centroid = (0.0,0.0)
        self.type = Shape.UNDEFINED
        self.bounder = (0,0,0,0)
        #
        self.estimate_shape() 
        self.compute_centroid() 

    def __repr__(self):
        return "[%d, %d]"%(self.position)

    def is_type(self, t):
        return t == self.type

    def compute_centroid(self):
        x, y, w, h = self.bounder 
        self.centroid = (
                x + w/2,
                y + h/2
                )
#        self.centroid = (
#                int(self.moments_["m10"] / self.moments_["m00"]),
#                int(self.moments_["m01"] / self.moments_["m00"])
#                )

    #def compute_area(self):
    #    self.area = cv2.contourArea(self.contour)
#        try:
#            self.area = self.moments_['m00']
#        except:
#            self.area = cv2.contourArea(self.contour)
    @property 
    def bound_rect(self):
        return self.bounder

    def estimate_shape(self):
        peri = cv2.arcLength(self.contour, True)
        approx = cv2.approxPolyDP(self.contour, 0.04 * peri, True)
        
        vertices = len(approx)
        self.bounder = cv2.boundingRect(approx)
        x,y,w,h = self.bounder
        self.area = w * h
        if self.area < 25:
            return

        if vertices == 3:
            self.type = Shape.TRIANGLE
        if vertices == 4:
            ar = w/float(h)
            #self.type = Shape.RECTANGLE
            #if 1 - 0.15 < math.fabs((peri / 4) ** 2) / self.area < 1 + 0.15:
            #    self.type = Shape.SQUARE
            self.type = Shape.SQUARE if (
                    self.SQUARE_RATIO_LIMITS[0]<=ar<=self.SQUARE_RATIO_LIMITS[1]
                    ) else Shape.RECTANGLE
        elif vertices == 5:
            self.type = Shape.POLYGON
        else:
            self.type = Shape.CIRCLE
    
    def highlight(self, ref_img, bColor=Color.GREEN, cColor=Color.WHITE):
        cv2.drawContours(ref_img, [self.contour], -1, bColor, 2)
        cv2.circle(ref_img, self.position, 1, cColor, -1)

    @property
    def position(self):
        return self.centroid
    @property
    def size(self):
        return self.area
    @property
    def ul_corner(self):
        x, y, w, h = self.bounder
        return (x, y)
    
    @property
    def dr_corner(self):
        x, y, w, h = self.bounder
        return (x + w, y + h)
   
    @property
    def ur_corner(self):
        x, y, w, h = self.bounder
        return (x + w, y)

    @property
    def r_edge(self):
        x, y, w, h = self.bounder
        return (x + w, y + h/2)
    
    @property
    def width(self):
        return self.bounder[2]
    @property
    def height(self):
        return self.bounder[3]
class ShapeList(list):
    def query(self, key):
        return ShapeList([x for x in self if x.type == key])

    def count(self, key):
        return len(self.query(key))

    def highlight(self, ref_img, border_color=Color.GREEN, center_color=Color.WHITE):
        assert ref_img is not None
        for i in self:
            i.highlight(ref_img, bColor=border_color, cColor=center_color)

    def remove(self, key):
        q = self.query(key)
        return ShapeList([x for x in self if x not in q])
    
    def get_bins_in_range(self, key, rng):
        q = self.query(key)
        return ShapeList([p for p in q if rng[0]<=p.position[0]<=rng[2] and rng[1]<=p.position[1]<=rng[3]])

    @property
    def min(self):
        return sorted(self, key=lambda x:x.size)[0]


