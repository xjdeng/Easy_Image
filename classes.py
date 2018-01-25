from __future__ import absolute_import
try:
    from . import detect
    from . import exif_json
except ImportError:
    import detect
    import exif_json
import cv2, dlib
import numpy as np

default_detector = detect.DetectorParams('dlib')

class EasyImage(object):
    
    def __init__(self, myinput):
        if isinstance(myinput, np.ndarray):
            self.path = None
            self._img = myinput
        else:
            raise(NotAnImage)
            
    def detect_faces(self, detector = default_detector):
        faces = detector.run(self)
        if len(faces) == 0:
            return []
        else:
            return [EasyFace(self, face) for face in faces]
            
    def getimg(self):
        return self._img

class EasyImageFile(EasyImage):
    
    def __init__(self, mypath):
        if isinstance(mypath, str):
            self.path = mypath
            self._img = cv2.imread(mypath)
        else:
            raise(NotAnImage)
    
    def force_detect_faces(self, detector = default_detector):
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
            
    
class EasyFace(EasyImage):

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

class NotFace(Exception):
    pass