from __future__ import absolute_import
import cv2, dlib
import classes
import os
from path import Path as path

mypath = os.path.abspath(__file__)
dir_path = os.path.dirname(mypath)
haarpath = dir_path + "/haarcascades/"
lbppath = dir_path + "/lbpcascades/"

def convert_rect(myinput0):
    myinput = [int(x) for x in myinput0]
    x = myinput[0]
    y = myinput[1]
    w = myinput[2]
    h = myinput[3]
    return dlib.rectangle(x,y, w + x, y + h)
    
def default_haar(img, minNeighbors = 5, scaleFactor = 1.1, *args, **kwargs):
    haar = haarpath + "haarcascade_frontalface_alt2.xml"
    return using_cascades(img, haar, minNeighbors = 5, scaleFactor = 1.1,\
                          *args, **kwargs)

def default_lbp(img, minNeighbors = 5, scaleFactor = 1.1, *args, **kwargs):
    lbp = lbppath + "lbpcascade_frontalface.xml"
    return using_cascades(img, lbp, minNeighbors = 5, scaleFactor = 1.1,\
                          *args, **kwargs) 
    
def get_gray(img):
    if isinstance(img, classes.EasyImage):
        img_ref = img.getimg()
    else:
        img_ref = cv2.imread(img)
        if img_ref is None:
            raise(classes.NotAnImage)
    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    return gray1

def haarcascades():
    hp = path(haarpath)
    return hp.files()

def lbpcascasdes():
    lp = path(lbppath)
    return lp.files()

def using_cascades(img, cascPath, minNeighbors = 5, scaleFactor = 1.1,\
                   *args, **kwargs):
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
        