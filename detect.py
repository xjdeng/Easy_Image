import cv2, dlib
import classes
import os
from path import Path as path

mypath = os.path.abspath(__file__)
dir_path = os.path.dirname(mypath)
haarpath = dir_path + "/haarcascades/"
lbppath = dir_path + "/lbpcascades/"

def convert_rect(myinput0):
    """
Converts the output from OpenCV's Cascade Classifier to an equivalent
dlib.rectangle.
    """
    myinput = [int(x) for x in myinput0]
    x = myinput[0]
    y = myinput[1]
    w = myinput[2]
    h = myinput[3]
    return dlib.rectangle(x,y, w + x, y + h)
    
def default_haar(img, minNeighbors = 5, scaleFactor = 1.1, *args, **kwargs):
    """
Detects the faces in an image using haarcascade_frontalface_alt2.xml Cascade.
The faces will be returned in a dlib.rectangles object (which is basically a
list of dlib.rectangle objects.)
    """
    haar = haarpath + "haarcascade_frontalface_alt2.xml"
    return using_cascades(img, haar, minNeighbors = 5, scaleFactor = 1.1,\
                          *args, **kwargs)

def default_lbp(img, minNeighbors = 5, scaleFactor = 1.1, *args, **kwargs):
    """
Detects the faces in an image using lbpcascade_frontalface.xml Cascade.
The faces will be returned in a dlib.rectangles object (which is basically a
list of dlib.rectangle objects.)
    """
    lbp = lbppath + "lbpcascade_frontalface.xml"
    return using_cascades(img, lbp, minNeighbors = 5, scaleFactor = 1.1,\
                          *args, **kwargs) 
    
def get_gray(img):
    """
Takes an image (either as an EasyImage object or path to an image) and
converts it to greyscale in numpy.ndarray format.

Not intended for the average end-user.
    """
    if isinstance(img, classes.EasyImage):
        img_ref = img.getimg()
    else:
        img_ref = cv2.imread(img)
        if img_ref is None:
            raise(classes.NotAnImage)
    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    return gray1

def haarcascades():
    """
Get a list of Haar Cascades included in the package. Not intended for use in
in the app you're building; just for your own information.
    """
    hp = path(haarpath)
    return hp.files()

def lbpcascasdes():
    """
Get a list of Lbp Cascades included in the package. Not intended for use in
in the app you're building; just for your own information.
    """
    lp = path(lbppath)
    return lp.files()

def using_cascades(img, cascPath, minNeighbors = 5, scaleFactor = 1.1,\
                   minSize = (0,0), maxSize = (0,0), *args, **kwargs):
    """
Detects the faces in an image using a Cascade on your local hard drive (cascPath.)
The faces will be returned in a dlib.rectangles object (which is basically a
list of dlib.rectangle objects.)
    """
    gray1 = get_gray(img)
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(gray1, minNeighbors = minNeighbors,\
                                         scaleFactor = scaleFactor, *args,\
                                         **kwargs)
    result = dlib.rectangles()
    for f in faces:
        result.append(convert_rect(f))
    return result
   
    

def using_dlib(img, level = 0):
    """
Returns a dlib.rectagles object of faces from img input.  The 'img' input can 
either be a classes.EasyImage object or a string path to an image.

If you can't find any or all of the faces in an image, try increasing the 
"level" parameter, although it'll increase execution time.
    """
    gray1 = get_gray(img)
    detector = dlib.get_frontal_face_detector()
    return detector(gray1, level)


def get_detector(mydict):
    """
Take an appropriately formatted dictionary and converts it to a DetectorParams
object. Basically the inverse of the DetectorParams.to_dict() method.
    """
    return DetectorParams(mydict['detector_type'], mydict['cascade_type'],\
                          mydict['cascade'], mydict['min_w'], mydict['min_h'], \
                          mydict['max_w'], mydict['max_h'], mydict['minNeighbors'], \
                          mydict['scaleFactor'], mydict['levels'])

class DetectorParams(object):
    """
This object holds the Face Detection parameters, including:
    
    - Which type (OpenCV's cascade or Dlib's hog)
    - The actual cascade if using OpenCV
    - The cascade's maxSize in (max_w, max_h) format
    - The cascade's minSize in (min_w, max_h) format
    - The cascade's minNeighbors
    - The cascade's scaleFactor
    - The dlib's levels.
    
You can ignore the cascade-related parameters if using dlib and vice versa.
    """
    def __init__(self, detector_type, cascade_type = None, cascade = None, \
                 min_w = 0, min_h = 0, max_w = 0, max_h = 0, minNeighbors=5, \
                 scaleFactor=1.1, levels=0):
        self.detector_type = detector_type
        self.cascade_type = cascade_type
        self.cascade = cascade
        self.min_w = min_w
        self.min_h = min_h
        self.max_w = max_w
        self.max_h = max_h
        self.minNeighbors = minNeighbors
        self.scaleFactor = scaleFactor
        self.levels = levels
    
    def run(self, img):
        """
Detect faces in img (EasyImage or EasyImageFile object) using the parameters
set in this DetectorParams object.
        """
        if (self.detector_type == "dlib") | (self.detector_type.lower() == 'hog'):
            return using_dlib(img, self.levels)
        else:
            if self.cascade_type == "haar":
                cascPath = haarpath
            else:
                cascPath = lbppath
            cascPath += self.cascade
            return using_cascades(img, cascPath, minSize = (self.min_w, self.min_h)\
                                  ,maxSize = (self.max_w, self.max_h), \
                                  minNeighbors = self.minNeighbors, \
                                  scaleFactor = self.scaleFactor)
    
    def to_dict(self):
        """
Converts this into a dictionary object (which can be converted to a JSON and
later stored in an image EXIF.)
        """
        output = {}
        output['detector_type'] = self.detector_type
        output['cascade_type'] = self.cascade_type
        output['cascade'] = self.cascade
        output['min_w'] = self.min_w
        output['min_h'] = self.min_h
        output['max_w'] = self.max_w
        output['max_h'] = self.max_h
        output['minNeighbors'] = self.minNeighbors
        output['scaleFactor'] = self.scaleFactor
        output['levels'] = self.levels
        return output
        