try:
    from . import exif_json
except ImportError:
    import exif_json
import copy, cv2, dlib, os
import numpy as np
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
    if isinstance(img, EasyImage):
        img_ref = img.getimg()
    else:
        img_ref = cv2.imread(img)
        if img_ref is None:
            raise(NotAnImage)
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

default_detector = DetectorParams('cascade','haar','haarcascade_frontalface_alt2.xml')

class EasyImage(object):
    """
Holds a generic image initialized as a numpy.ndarray (like what you'd get if 
you were to run cv2.imread on a path to an image.)
    """
    
    def __init__(self, myinput):
        if isinstance(myinput, np.ndarray):
            self.path = None
            self._img = myinput
        else:
            raise(NotAnImage)
            
    def detect_faces(self, detector = default_detector):
        """
Returns a list of faces detected in the image; each face is represented using 
an EasyFace object.
        """
        faces = detector.run(self)
        if len(faces) == 0:
            return []
        else:
            return [EasyFace(self, face) for face in faces]

    def draw_faces(self, detector = default_detector, color = (0, 255, 0),\
                   width = 2):
        """
Detects faces in an image then creates a new image with green rectangles 
drawn around the faces (at least in the default option.)
        """
        faces = self.detect_faces(detector = detector)
        nimg = copy.deepcopy(self._img)
        for f in faces:
            f0 = f.face
            x = f0.left()
            y = f0.top()
            x2 = f0.right()
            y2 = f0.bottom()
            cv2.rectangle(nimg, (x,y), (x2,y2), color, width)
        return EasyImage(nimg)
            
    def getimg(self):
        """
Returns the image stored in the object in numpy.ndarray format.
        """
        return self._img
    
    def save(self,newpath):
        """
Save the image at the new path: newpath
        """
        return cv2.imwrite(newpath, self._img)

class EasyImageFile(EasyImage):
    """
This is a subclass of EasyImage.. these are images that have been loaded from 
the local disk.  The self.path variable retains the path to the image.
    """
    
    def __init__(self, mypath):
        if isinstance(mypath, str):
            self.path = mypath
            self._img = cv2.imread(mypath)
        else:
            raise(NotAnImage)
            
    def detect_faces(self, cvv = None, dlibv = None, detector = default_detector):
        """
The all-purpose facial detection algorithm for image files. First, it'll search
for faces already cached in the EXIF data and if found, return those. 
Otherwise, it'll detect faces using the 'detector' specified. The default
detector uses Dlib's HOG detectors but the detector can be modified to use
HAAR or LBP classifiers. For more info, see the DetectorParams class in
detect.py.

The cvv and dlibv variables specify the versions of OpenCV and Dlib to look for
in the EXIF data. If they're specified as something other than None, then it'll
ignore the faces in the EXIF if the versions don't match.
        """
        test = self.faces_from_exif(cvv, dlibv, detector)
        if test is None:
            return []
        elif len(test) == 0:
            return self.force_detect_faces(detector)
        else:
            return test
            
    
    def faces_from_exif(self, cvv = None, dlibv = None, detector = None):
        """
Look for faces stored in the image EXIF. The cvv and dlibv variables specify
the versions of OpenCV and Dlib to look for in the EXIF data and the detector
specifies the detector to look for.  If any of those parameters are set to 
none, then it'll be ignored. If a particular parameter is not None and it
doesn't match what's stored in the EXIF, then the faces in the EXIF are ignored
and this will return [].

If the EXIF specifically says there are no faces in the image, then None is
returned.
        """
        if isinstance(detector, DetectorParams):
            detector = detector.to_dict()
        test = exif_json.load(self.path)
        if (test is None) or (isinstance(test, dict) == False):
            return []
        for x in ['OpenCV Version', 'Dlib Version', 'detector', 'faces']:
            if x not in test.keys():
                return []
        if (cvv is not None) & (cvv != test['OpenCV Version']):
            return []
        if (dlibv is not None) & (dlibv != test['Dlib Version']):
            return []
        if (detector is not None) & (detector != test['detector']):
            return []
        if len(test['faces']) == 0:
            return None
        return [EasyFace(self, dlib.rectangle(f[0],f[1],f[2],f[3])) for f in\
                test['faces']]
    
    def force_detect_faces(self, detector = default_detector):
        """
Detects faces in an image regardless whether faces are cached in the EXIF. If 
faces are found, store them in the EXIF, overwriting them if necessary. Also
stores the OpenCV and Dlib versions as well as the detector used to detect
the faces.
        """
        faces = super(EasyImageFile, self).detect_faces(detector = detector)
        output = {}
        output['OpenCV Version'] = cv2.__version__
        output['Dlib Version'] = dlib.__version__
        output['detector'] = detector.to_dict()
        if len(faces) > 0:
            output['faces'] = [[x.left(), x.top(), x.right(), x.bottom()] \
                  for x in [y.face for y in faces]]
        else:
            output['faces'] = []
        exif_json.save(self.path, output)
        return faces

    def remove_faces(self):
        """
Removes faces from the image's EXIF data        
        """
        exif_json.save(self.path, None)
            
    
class EasyFace(EasyImage):
    """
This is a special case of an EasyImage that's a face. It keeps pointers to the
original image that the face was detected from and a dlib.rectangle object
representing the faces as its constructors.
    """

    def __init__(self, an_easy_image, a_rect):
        if isinstance(an_easy_image, EasyImage) & \
        isinstance(a_rect, dlib.rectangle):
            self.parent_image = an_easy_image
            self.face = a_rect #should be class dlib.rectangle
        else:
            raise(NotFace)
    
    def detect_faces(self):
        return [self]
    
    def getimg(self):
        x = self.face.left()
        y = self.face.top()
        x2 = self.face.right()
        y2 = self.face.bottom()
        return self.parent_image.getimg()[y:y2,x:x2]
            
class NotAnImage(Exception):
    pass
#TODO: Add more functionality and information to this exception

class NotFace(Exception):
    pass
#TODO: Add more functionality and information to this exception        