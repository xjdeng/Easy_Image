try:
    from .detect import DetectorParams, EasyImage, EasyImageFile, EasyFace,\
    faces_in_dir, load_image
    from .detect import NotAnImage, NotFace
except ImportError:
    from detect import DetectorParams, EasyImage, EasyImageFile, EasyFace,\
    faces_in_dir, load_image
    from detect import NotAnImage, NotFace