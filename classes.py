import cv2
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
    
                

class NotAnImage(Exception):
    pass