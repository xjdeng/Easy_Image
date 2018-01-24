import cv2, dlib
import numpy as np

class EasyImage(object):
    
    def __init__(self, myinput):
        if isinstance(myinput, np.ndarray):
            self.path = None
            self._img = myinput
        else:
            raise(NotAnImage)
            
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
            
    #TODO: overload the getimg() function to return the face from self.face

class NotAnImage(Exception):
    pass

class NotFace(Exception):
    pass