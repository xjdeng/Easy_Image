try:
    from . import detect
    from . import exif_json
except ImportError:
    import detect
    import exif_json
import copy, cv2, dlib
import numpy as np

default_detector = detect.DetectorParams('dlib')

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
        if isinstance(detector, detect.DetectorParams):
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