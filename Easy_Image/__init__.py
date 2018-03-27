try:
    from .detect import DetectorParams, EasyImage, EasyImageFile, EasyFace,\
    faces_in_dir, load_image
    from . import age_and_gender
    from .detect import NotAnImage, NotFace
except ImportError:
    from detect import DetectorParams, EasyImage, EasyImageFile, EasyFace,\
    faces_in_dir, load_image
    import age_and_gender
    from detect import NotAnImage, NotFace