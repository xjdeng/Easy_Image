from tkinter import filedialog

try:
    from . import detect
except ImportError:
    import detect

def load():
    imgfile = filedialog.askopenfilename(initialdir = "/",\
                                         title = "Select Image:")
    return detect.EasyImageFile(imgfile)

def load_dir(recursive = False):
    imgdir = filedialog.askdirectory(initialdir = "/",\
                                         title = "Select Directory:")
    return detect.load_image_dir(imgdir, recursive = recursive)