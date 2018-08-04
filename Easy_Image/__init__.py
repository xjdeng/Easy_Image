try:
    from .detect import DetectorParams, EasyImage, EasyImageFile, EasyFace,\
    faces_in_dir, load_image, load_image_dir, EasyImageList, \
    EasyImageFileList, EasyFaceList, ImageFileList
    from .detect import NotAnImage, NotFace
    from . import gui
except ImportError:
    from detect import DetectorParams, EasyImage, EasyImageFile, EasyFace,\
    faces_in_dir, load_image, load_image_dir, EasyImageList, \
    EasyImageFileList, EasyFaceList, ImageFileList
    from detect import NotAnImage, NotFace
    import gui