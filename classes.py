from __future__ import absolute_import
import detect
import cv2, dlib
import numpy as np

detector = detect.default_haar

class EasyImage(object):
    
    def __init__(self, myinput):
        if isinstance(myinput, np.ndarray):
            self.path = None
            self._img = myinput
        else:
            raise(NotAnImage)
            
    def detect_faces(self):
        faces = detector(self)
        if len(faces) == 0:
            return None
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
    
class EasyFace(EasyImage):

    def __init__(self, an_easy_image, a_rect):
        if isinstance(an_easy_image, EasyImage) & \
        isinstance(a_rect, dlib.rectangle):
            self.parent_image = an_easy_image
            self.face = a_rect #should be class dlib.rectangle
        else:
            raise(NotFace)
    
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