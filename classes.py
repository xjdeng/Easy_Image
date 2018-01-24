import cv2
import numpy as np

class EasyImage(object):
    
    def __init__(self, myinput):
        if isinstance(myinput, str):
            self.path = str
            self._img = cv2.imread(myinput)
        elif isinstance(myinput, np.ndarray):
            self.path = None
            self._img = myinput
        else:
            raise(NotAnImage)
            
    def getimg(self):
        return self._img
            

class NotAnImage(Exception):
    pass