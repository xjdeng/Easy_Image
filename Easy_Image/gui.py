from tkinter import filedialog

try:
    from . import detect
except ImportError:
    import detect

def load():
    imgfile = filedialog.askopenfilename(initialdir = "/",\
                                         title = "Select Image:")
    return detect.EasyImageFile(imgfile)