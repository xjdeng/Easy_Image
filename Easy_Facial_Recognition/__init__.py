try:
    from .detect import DetectorParams, EasyImage, EasyImageFile, EasyFace, faces_in_dir
    from .detect import NotAnImage, NotFace
except ImportError:
    from detect import DetectorParams, EasyImage, EasyImageFile, EasyFace, faces_in_dir
    from detect import NotAnImage, NotFace