from __future__ import absolute_import
import cv2, dlib
import classes

def using_dlib(img, level = 0):
    """
Returns a dlib.rectagles object of faces from img input.  The 'img' input can 
either be a classes.EasyImage object or a string path to an image.

If you can't find any or all of the faces in an image, try increasing the 
"level" parameter, although it'll increase execution time.
    """
    if isinstance(img, classes.EasyImage):
        img_ref = img.img
    else:
        img_ref = cv2.imread(img)
        if img_ref is None:
            raise(classes.NotAnImage)
    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    return detector(gray1, level)
        