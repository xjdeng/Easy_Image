try:
    from .detect import DetectorParams, EasyImage, EasyImageFile, EasyFace
    from .detect import NotAnImage, NotFace
except ImportError:
    from detect import DetectorParams, EasyImage, EasyImageFile, EasyFace
    from detect import NotAnImage, NotFace