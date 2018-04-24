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
        self.compute_centroid() 
        self.estimate_shape() 
        self.compute_area()

    def __repr__(self):
        return "[%d, %d]"%(self.position)

    def is_type(self, t):
        return t == self.type

    def compute_centroid(self):
        self.centroid = (
                int(self.moments_["m10"] / self.moments_["m00"]),
                int(self.moments_["m01"] / self.moments_["m00"])
                )

    def compute_area(self):
        self.area = self.moments_['m00']
    @property 
    def bound_rect(self):
        return self.bounder

    def estimate_shape(self):
        peri = cv2.arcLength(self.contour, True)
        approx = cv2.approxPolyDP(self.contour, 0.04 * peri, True)
        self.type = Shape.UNDEFINED 
        vertices = len(approx)
        self.bounder = cv2.boundingRect(approx)

        if vertices == 3:
            self.type = Shape.TRIANGLE
        if vertices == 4:
            x,y,w,h = self.bounder
            ar = w/float(h)
            self.type = Shape.SQUARE if (
                    self.SQUARE_RATIO_LIMITS[0]<=ar<=self.SQUARE_RATIO_LIMITS[1]
                    ) else Shape.RECTANGLE
        elif vertices == 5:
            self.type = Shape.POLYGON
        else:
            self.type = Shape.CIRCLE
    
    def highlight(self, ref_img):
        cv2.drawContours(ref_img, [self.contour], -1, Color.GREEN, 2)
        cv2.circle(ref_img, self.position, 1, Color.WHITE, -1)

    @property
    def position(self):
        return self.centroid
    @property
    def size(self):
        return self.area

class ShapeList(list):
    def query(self, key):
        return ShapeList([x for x in self if x.type == key])

    def count(self, key):
        return len(self.query(key))

    def highlight(self, ref_img):
        assert ref_img is not None
        for i in self:
            i.highlight(ref_img)

    @property
    def min(self):
        return sorted(self, key=lambda x:x.size)[0]


