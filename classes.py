import cv2
import numpy as np

class EasyImage(object):
    
    def __init__(self, myinput):
        if isinstance(myinput, str):
            self.path = str
            self.img = cv2.imread(myinput)
        elif isinstance(myinput, np.ndarray):
            self.path = None
            self.img = myinput
        else:
            raise(NotAnImage)
            

class NotAnImage(Exception):
    pass