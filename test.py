from path import Path as path
import os,time

try:
    from . import detect
except ImportError:
    import detect

mypath = os.path.abspath(__file__)
dirpath = os.path.dirname(mypath)
testdir = dirpath + "/tests/"
destination_dir = dirpath + "/test_results/"


def benchmark(testimgs = testdir, destination = destination_dir, detector = \
              detect.default_detector):
    print("Please don't switch to any other programs while your benchmark is running")
    test_dir = path(testdir)
    destination_dir = path(destination)
    if destination_dir.exists() == False:
        destination_dir.mkdir()
    else:
        for f in destination_dir.files():
            f.remove()
    t0 = time.time()
    for f in test_dir.files():
        try:
            testimg = detect.EasyImageFile(f)
            testimg.remove_faces()
            newimg = testimg.draw_faces(detector)
            newimg.save(destination_dir + "/" + str(f.name))
        except detect.NotAnImage:
            pass
    t1 = time.time()
    timefile = open(destination_dir + "/results.txt", 'w')
    timefile.write(str(t1 - t0) + "\n" + str(detector.to_dict()))

def test1():
    det = detect.DetectorParams('dlib')
    benchmark(detector=det, destination="test_results2")

def test2():
    det = detect.DetectorParams('cascade','haar','haarcascade_frontalface_alt2.xml')
    benchmark(detector=det, destination="test_results2")
    
def test3():
    det = detect.DetectorParams('cascade','lbp','lbpcascade_frontalface.xml')
    benchmark(detector=det, destination="test_results3")
    