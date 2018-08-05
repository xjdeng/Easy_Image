try:
    from . import exif_json
    from . import compare
    from . import classify_pytorch as classify
    from . import colordescriptor as cd
except ImportError:
    import exif_json
    import compare
    import colordescriptor as cd
    try:
        import classify_pytorch as classify
    except ImportError:
        print("Warning: Pytorch not found. You will not be able to classify images!")
import copy, cv2, dlib, os
import numpy as np
from path import Path as path
from imutils import face_utils
import face_recognition_models as frm
from skimage.io import imread
from urllib.error import URLError, HTTPError
import warnings
import random
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", message="Unverified HTTPS request is being made")

mypath = os.path.abspath(__file__)
dir_path = os.path.dirname(mypath)
haarpath = dir_path + "/haarcascades/"
lbppath = dir_path + "/lbpcascades/"

default_predictor = dlib.shape_predictor(frm.pose_predictor_model_location())

classify_field = 50706
imagenet_model = 'squeezenet'

ORB = cv2.ORB_create()

def convert_rect(myinput0):
    """
Converts the output from OpenCV's Cascade Classifier to an equivalent
dlib.rectangle.
    """
    myinput = [int(x) for x in myinput0]
    x = myinput[0]
    y = myinput[1]
    w = myinput[2]
    h = myinput[3]
    return dlib.rectangle(x,y, w + x, y + h)
    
def default_haar(img, minNeighbors = 5, scaleFactor = 1.1, *args, **kwargs):
    """
Detects the faces in an image using haarcascade_frontalface_alt2.xml Cascade.
The faces will be returned in a dlib.rectangles object (which is basically a
list of dlib.rectangle objects.)
    """
    haar = haarpath + "haarcascade_frontalface_alt2.xml"
    return using_cascades(img, haar, minNeighbors = 5, scaleFactor = 1.1,\
                          *args, **kwargs)

def default_lbp(img, minNeighbors = 5, scaleFactor = 1.1, *args, **kwargs):
    """
Detects the faces in an image using lbpcascade_frontalface.xml Cascade.
The faces will be returned in a dlib.rectangles object (which is basically a
list of dlib.rectangle objects.)
    """
    lbp = lbppath + "lbpcascade_frontalface.xml"
    return using_cascades(img, lbp, minNeighbors = 5, scaleFactor = 1.1,\
                          *args, **kwargs)

def get_gray(img):
    """
Takes an image (either as an EasyImage object or path to an image) and
converts it to greyscale in numpy.ndarray format.

Not intended for the average end-user.
    """
    if isinstance(img, EasyImage):
        img_ref = img.getimg()
    else:
        img_ref = cv2.imread(img)
        if img_ref is None:
            raise(NotAnImage)
    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    return gray1

def haarcascades():
    """
Get a list of Haar Cascades included in the package. Not intended for use in
in the app you're building; just for your own information.
    """
    hp = path(haarpath)
    return hp.files()

def lbpcascasdes():
    """
Get a list of Lbp Cascades included in the package. Not intended for use in
in the app you're building; just for your own information.
    """
    lp = path(lbppath)
    return lp.files()

def load_image(input1):
    if isinstance(input1, str):
        if input1.startswith("http"):
            return EasyImage(input1)
        else:
            return EasyImageFile(input1)
    elif isinstance(input1, np.ndarray):
        return EasyImage(input1)
    elif isinstance(input1, EasyImage):
        return copy.copy(input1)
    else:
        raise(NotAnImage)

def get_all_files(folder):
    f = path(folder)
    folders = f.dirs()
    files = f.files()
    result = files
    for i in folders:
        result += get_all_files(i)
    return result

def load_image_dir(mydir, recursive = False, maximgs = None, strout = False):
    if recursive == True:
        files = get_all_files(mydir)
    else:
        files = path(mydir).files()
    random.shuffle(files)
    if strout == False:
        images = EasyImageFileList()
    else:
        images = ImageFileList()
    i = 0
    if maximgs == None:
        num = len(files)
    else:
        num = maximgs
    while (len(images) < num) & (i < len(files)):
        test = files[i]
        i += 1
        try:
            tmp = EasyImageFile(test)
            if strout == True:
                images.append(str(tmp.path))
            else:
                images.append(tmp)
        except NotAnImage:
            pass
    return images

def from_urls(urls):
    images = EasyImageList()
    for u in urls:
        try:
            img = EasyImage(u)
            images.append(img)
        except NotAnImage:
            pass
    return images

def using_cascades(img, cascPath, minNeighbors = 5, scaleFactor = 1.1,\
                   minSize = (0,0), maxSize = (0,0), *args, **kwargs):
    """
Detects the faces in an image using a Cascade on your local hard drive (cascPath.)
The faces will be returned in a dlib.rectangles object (which is basically a
list of dlib.rectangle objects.)
    """
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


def get_detector(mydict):
    """
Take an appropriately formatted dictionary and converts it to a DetectorParams
object. Basically the inverse of the DetectorParams.to_dict() method.
    """
    return DetectorParams(mydict['detector_type'], mydict['cascade_type'],\
                          mydict['cascade'], mydict['min_w'], mydict['min_h'], \
                          mydict['max_w'], mydict['max_h'], mydict['minNeighbors'], \
                          mydict['scaleFactor'], mydict['levels'])

class DetectorParams(object):
    """
This object holds the Face Detection parameters, including:
    
    - Which type (OpenCV's cascade or Dlib's hog)
    - The actual cascade if using OpenCV
    - The cascade's maxSize in (max_w, max_h) format
    - The cascade's minSize in (min_w, max_h) format
    - The cascade's minNeighbors
    - The cascade's scaleFactor
    - The dlib's levels.
    
You can ignore the cascade-related parameters if using dlib and vice versa.
    """
    def __init__(self, detector_type, cascade_type = None, cascade = None, \
                 min_w = 0, min_h = 0, max_w = 0, max_h = 0, minNeighbors=5, \
                 scaleFactor=1.1, levels=0):
        self.detector_type = detector_type
        self.cascade_type = cascade_type
        self.cascade = cascade
        self.min_w = min_w
        self.min_h = min_h
        self.max_w = max_w
        self.max_h = max_h
        self.minNeighbors = minNeighbors
        self.scaleFactor = scaleFactor
        self.levels = levels
    
    def run(self, img):
        """
Detect faces in img (EasyImage or EasyImageFile object) using the parameters
set in this DetectorParams object.
        """
        if (self.detector_type == "dlib") | (self.detector_type.lower() == 'hog'):
            return using_dlib(img, self.levels)
        else:
            if self.cascade_type == "haar":
                cascPath = haarpath
            else:
                cascPath = lbppath
            cascPath += self.cascade
            return using_cascades(img, cascPath, minSize = (self.min_w, self.min_h)\
                                  ,maxSize = (self.max_w, self.max_h), \
                                  minNeighbors = self.minNeighbors, \
                                  scaleFactor = self.scaleFactor)
    
    def to_dict(self):
        """
Converts this into a dictionary object (which can be converted to a JSON and
later stored in an image EXIF.)
        """
        output = {}
        output['detector_type'] = self.detector_type
        output['cascade_type'] = self.cascade_type
        output['cascade'] = self.cascade
        output['min_w'] = self.min_w
        output['min_h'] = self.min_h
        output['max_w'] = self.max_w
        output['max_h'] = self.max_h
        output['minNeighbors'] = self.minNeighbors
        output['scaleFactor'] = self.scaleFactor
        output['levels'] = self.levels
        return output

default_detector = DetectorParams('cascade','haar','haarcascade_frontalface_alt2.xml')

def verify_img(img):
    if img is None:
        raise(NotAnImage)
    if np.max(img) < 1:
        img2 = 255*img
        img2 = img2.astype('uint8')
        return img2
    else:
        return img.astype('uint8')

class EasyImage(object):
    """
Holds a generic image initialized as a numpy.ndarray (like what you'd get if 
you were to run cv2.imread on a path to an image.)

You can also pass a URL starting with http into the input and it'll download
and load that image.
    """
    
    def __init__(self, myinput):
        if isinstance(myinput, np.ndarray):
            self.path = None
            self._img = verify_img(myinput)
        elif (isinstance(myinput, str)) or (isinstance(myinput, bytes)):
            if isinstance(myinput, bytes):
                myinput = myinput.decode('ascii','ignore')
            if myinput.startswith("http"):
                try:
                    img = imread(myinput)
                    self._img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    self.path = None
                except (URLError, HTTPError, AttributeError):
                    raise(NotAnImage)
            else:
                raise(NotAnImage)
        else:
            raise(NotAnImage)
    
    def classify(self, mod = imagenet_model):
        """
Classifies the image using an ImageNet model (default: squeezenet) by returning
a list of tuples of classifications and their respective probabilities.

This require Pytorch 0.3.0
        """

        return classify.classify(self.getimg())
    
    def describe(self, bins = (8,12,3)):
        """
Get a color histogram for the current image, used to "summarize" it. For a 
full discussion on the bins and the theory behind this, please see:
https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
        """
        mycd = cd.ColorDescriptor(bins)
        return mycd.describe(self.getimg())
                
    def detect_faces(self, detector = default_detector):
        """
Returns a list of faces detected in the image; each face is represented using 
an EasyFace object.
        """
        faces = detector.run(self)
        if len(faces) == 0:
            return EasyFaceList()
        else:
            return EasyFaceList([EasyFace(self, face) for face in faces])

    def draw_faces(self, detector = default_detector, color = (0, 255, 0),\
                   width = 2):
        """
Detects faces in an image then creates a new image with green rectangles 
drawn around the faces (at least in the default option.)
        """
        faces = self.detect_faces(detector = detector)
        nimg = copy.deepcopy(self._img)
        for f in faces:
            f0 = f.face
            x = f0.left()
            y = f0.top()
            x2 = f0.right()
            y2 = f0.bottom()
            cv2.rectangle(nimg, (x,y), (x2,y2), color, width)
        return EasyImage(nimg)
            
    def getimg(self):
        """
Returns the image stored in the object in numpy.ndarray format.
        """
        return self._img
    
    def ORB(self):
        """
Extracts the image's features using the ORB algorithm
        """
        #https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html
        grey = get_gray(self)
        kp0 = ORB.detect(grey,None)
        _, des = ORB.compute(grey, kp0)
        return des
    
    def plot(self):
        """
Like show() but plots the image using Matplotlib
        """
        plt.imshow(cv2.cvtColor(self.getimg(), cv2.COLOR_BGR2RGB))
    
    def resize(self, width, height, inplace = True):
        """
Resizes the image to the specified width and height
        """
        try:
            newimg = cv2.resize(self._img, (width, height))
        except AttributeError:
            if inplace == True:
                raise(InvalidOperation)
            else:
                return EasyImage(cv2.resize(self.getimg(), (width, height)))
        if inplace == True:
            self._img = newimg
            return self
        else:
            return EasyImage(newimg)
    
    def save(self,newpath):
        """
Save the image at the new path: newpath
        """
        return cv2.imwrite(newpath, self._img)
    
    def show(self, flag = cv2.WINDOW_NORMAL):
        """
Displays the image using OpenCV's imshow
        """
        #https://github.com/opencv/opencv/issues/7343
        img = self.getimg()
        wname = str(hash(img.tostring()))
        cv2.namedWindow(wname, flag) 
        cv2.imshow(wname, img)
        key = cv2.waitKey()
        cv2.destroyWindow(wname)
        return key
    
    def signature(self, width = 30, height = 30):
        """
Get a numpy signature of the image, which is its resized image flattened.
        """
        tmp = self.resize(width, height, False)
        return tmp.getimg().flatten()

class EasyImageFile(EasyImage):
    """
This is a subclass of EasyImage.. these are images that have been loaded from 
the local disk.  The self.path variable retains the path to the image.
    """
    
    def __init__(self, mypath):
        if isinstance(mypath, str):
            self.path = mypath
            self._img = verify_img(cv2.imread(mypath))
        else:
            raise(NotAnImage)

    def classify(self, mod = imagenet_model):
        """
Returns a dictionary with the probabilities that the image of is a particular
type. It tries to search for classifications in the EXIF data before finally
doing the classification, in order to save time and computing power.        
        """
        test = self.classify_from_exif(mod)
        if test is None:
            return []
        elif len(test) == 0:
            return self.classify_forced(mod)
        else:
            return test

    
    def classify_from_exif(self, mod = imagenet_model):
        """
Only get the classifications of the image stored in the EXIF data. If not
found, return an empty dict. If error, return None. Used primarily with classify()
        """
        test = exif_json.load(self.path, classify_field)
        if (test is None) or (isinstance(test, list) == False):
            return {}
        elif len(test) != 2:
            return {}
        elif mod != test[0]:
            return {}
        elif isinstance(test[1], dict) == False:
            return {}
        elif len(test[1]) == 0:
            return None
        else:
            return test[1]
        
    
    def classify_forced(self, mod = imagenet_model):
        """
Ignores the classifications stored in the EXIF and forces a run of the 
image classification algorithm.     
        """
        classes = super(EasyImageFile, self).classify(mod)
        output = [mod, classes]
        exif_json.save(self.path, output, classify_field)
        return classes
            
    def detect_faces(self, cvv = None, dlibv = None, detector = default_detector):
        """
The all-purpose facial detection algorithm for image files. First, it'll search
for faces already cached in the EXIF data and if found, return those. 
Otherwise, it'll detect faces using the 'detector' specified. The default
detector uses Dlib's HOG detectors but the detector can be modified to use
HAAR or LBP classifiers. For more info, see the DetectorParams class in
detect.py.

The cvv and dlibv variables specify the versions of OpenCV and Dlib to look for
in the EXIF data. If they're specified as something other than None, then it'll
ignore the faces in the EXIF if the versions don't match.
        """
        test = self.faces_from_exif(cvv, dlibv, detector)
        if test is None:
            return EasyFaceList()
        elif len(test) == 0:
            return self.force_detect_faces(detector)
        else:
            return test
            
    
    def faces_from_exif(self, cvv = None, dlibv = None, detector = None):
        """
Look for faces stored in the image EXIF. The cvv and dlibv variables specify
the versions of OpenCV and Dlib to look for in the EXIF data and the detector
specifies the detector to look for.  If any of those parameters are set to 
none, then it'll be ignored. If a particular parameter is not None and it
doesn't match what's stored in the EXIF, then the faces in the EXIF are ignored
and this will return [].

If the EXIF specifically says there are no faces in the image, then None is
returned.
        """
        if isinstance(detector, DetectorParams):
            detector = detector.to_dict()
        test = exif_json.load(self.path)
        if (test is None) or (isinstance(test, dict) == False):
            return EasyFaceList()
        for x in ['OpenCV Version', 'Dlib Version', 'detector', 'faces']:
            if x not in test.keys():
                return EasyFaceList()
        if (detector is not None) & (detector != test['detector']):
            return EasyFaceList()
        if len(test['faces']) == 0:
            return None
        return EasyFaceList([EasyFace(self, dlib.rectangle(f[0],f[1],f[2],\
                                                           f[3])) for f in\
                test['faces']])
    
    def force_detect_faces(self, detector = default_detector):
        """
Detects faces in an image regardless whether faces are cached in the EXIF. If 
faces are found, store them in the EXIF, overwriting them if necessary. Also
stores the OpenCV and Dlib versions as well as the detector used to detect
the faces.
        """
        faces = super(EasyImageFile, self).detect_faces(detector = detector)
        output = {}
        output['OpenCV Version'] = cv2.__version__
        output['Dlib Version'] = dlib.__version__
        output['detector'] = detector.to_dict()
        if len(faces) > 0:
            output['faces'] = [[x.left(), x.top(), x.right(), x.bottom()] \
                  for x in [y.face for y in faces]]
        else:
            output['faces'] = []
        exif_json.save(self.path, output)
        return faces

    def remove_exif(self):
        """
Removes images' exif date including faces and classifications        
        """
        exif_json.save(self.path, None)
            
    
class EasyFace(EasyImage):
    """
This is a special case of an EasyImage that's a face. It keeps pointers to the
original image that the face was detected from and a dlib.rectangle object
representing the faces as its constructors.
    """

    def __init__(self, an_easy_image, a_rect):
        if isinstance(an_easy_image, EasyImage) & \
        isinstance(a_rect, dlib.rectangle):
            self.parent_image = an_easy_image
            self.face = a_rect #should be class dlib.rectangle
        else:
            raise(NotFace)
            
    def compare_face(self, face, threshold = 0.5):
        """
Compares 2 faces, seeing whether they're the same person. If you want the actual
distance between them, set threshold = None.
        """
        encoding1 = compare.face_encodings(self.parent_image.getimg(), \
                                           [self.face])[0]
        encoding2 = compare.face_encodings(face.parent_image.getimg(), \
                                           [face.face])[0]
        if threshold is None:
            return compare.face_distance([encoding1], encoding2)[0]
        else:
            return compare.compare_faces([encoding1], encoding2, threshold)[0] 
    
    def detect_faces(self):
        return EasyFaceList([self])
       
    def getimg(self):
        x = self.face.left()
        y = self.face.top()
        x2 = self.face.right()
        y2 = self.face.bottom()
        return self.parent_image.getimg()[y:y2,x:x2]

    def getpoints(self, predictor = None):
        """
    NEW Function: take a face and a predictor object and returns
    a list of tuples containing coordinates of the boundaries on the face.
    
    If you don't have a predictor, then create one:
        
    import dlib
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    Note that you can download shape_predictor_68_face_landmarks.dat from a lot of
    places; just Google for one.
        """
        if predictor is None:
            predictor = default_predictor
        gray1 = cv2.cvtColor(self.parent_image.getimg(), cv2.COLOR_BGR2GRAY)
        shape1 = predictor(gray1, self.face)
        points1 = face_utils.shape_to_np(shape1)
        return list(map(tuple, points1))
            
class NotAnImage(Exception):
    pass
#TODO: Add more functionality and information to this exception

class NotFace(Exception):
    pass
#TODO: Add more functionality and information to this exception

class InvalidOperation(Exception):
    pass
#TODO: Add more functionality to this exception

class EasyImageList(list):
    """
This object extends the Python list object but makes it easy to do mass 
EasyImage operations on all of them like detect_faces(). Other functions
will be implemented in the future.
    """
    
    def __init__(self, x = []):
        for a in x:
            self.append(a)
    
    def __add__(self, x):
        if isinstance(x, EasyImageList):
            return EasyImageList(super(EasyImageList, self).__add__(x))
    
    def __iadd__(self, x):
        if isinstance(x, EasyImageList):
            return super(EasyImageList, self).__iadd__(x)
    
    def append(self, x):
        if isinstance(x, EasyImage):
            super(EasyImageList, self).append(x)
            
    def classify(self, mod = imagenet_model):
        return [img.classify(mod = mod) for img in self]
    
    def classify_list(self, mod = imagenet_model):
        classes = self.classify(mod = mod)
        return classify.combine_classifications(classes)
    
    def cluster(self, n_clusters, debug = False):
        tmp = self[0].signature()
        g = np.zeros((len(self), len(tmp)))
        for i, img in enumerate(self):
            g[i,:] = img.signature()
        g = g / 255.0       
        model = KMeans(n_clusters = n_clusters)
        clusters = model.fit_predict(g)
        n_clusters = max(0, max(clusters))
        results = []
        for i in range(0, n_clusters + 1):
            results.append(self.__class__())
        for i,c in enumerate(clusters):
            results[c].append(self[i])
        if debug == True:
            return (results, clusters, model)
        else:
            return results
    
    def cluster_smart(self, min_clusters = 2, max_clusters = None, debug = False):
        if max_clusters is None:
            max_clusters = len(self) - 2
        tmp = self[0].signature()
        g = np.zeros((len(self), len(tmp)))
        for i, img in enumerate(self):
            g[i,:] = img.signature()
        g = g / 255.0
        best_n, best_s = (1, -2)
        scores = []
        for i in range(min_clusters, max_clusters + 1):
            model = KMeans(n_clusters = i)
            test = model.fit_predict(g)
            score = silhouette_score(g, test)
            if score > best_s:
                best_n, best_s = (i, score)
                if debug == True:
                    scores.append((i, score))
        model = KMeans(n_clusters = best_n)
        clusters = model.fit_predict(g)
        n_clusters = max(0, max(clusters))
        results = []
        for i in range(0, n_clusters + 1):
            results.append(self.__class__())
        for i,c in enumerate(clusters):
            results[c].append(self[i])
        if debug == True:
            return (results, clusters, model, scores)
        else:
            return results
    
    def cluster_gmm_smart(self, min_clusters = 2, max_clusters = None, \
                          debug = False):
        """
BETA. Very slow and inaccurate right now.
        """
        if max_clusters is None:
            max_clusters = len(self) - 2
        tmp = self[0].signature()
        g = np.zeros((len(self), len(tmp)))
        for i, img in enumerate(self):
            g[i,:] = img.signature()
        g = g / 255.0
        best_n, best_s = (1, 999999999)
        for i in range(min_clusters, max_clusters + 1):
            model = GMM(n_components = i)
            model.fit(g)
            score = model.bic(g)
            if score < best_s:
                best_n, best_s = (i, score)
        model = GMM(n_components = best_n)
        model.fit(g)
        clusters = model.predict(g)
        n_clusters = max(0, max(clusters))
        results = []
        for i in range(0, n_clusters + 2):
            results.append(self.__class__())
        for i,c in enumerate(clusters):
            results[c].append(self[i])
        if debug == True:
            return (results, clusters, model)
        else:
            return results        
            
    def detect_faces(self, detector = default_detector):
        faces = [i.detect_faces(detector) for i in self]
        tmp = EasyFaceList()
        for f in faces:
            tmp += EasyFaceList(f)
        return tmp
    
    def pre_keras(self):
        output = np.expand_dims(self[0].getimg(), axis=0)
        for i in range(1, len(self)):
            output = np.vstack((output, np.expand_dims(self[i].getimg(), \
                                                       axis=0)))
        return output
    
    def resize(self, width, height, inplace = True):
        if inplace == True:
            for img in self:
                img.resize(width, height, True)
            return self
        else:
            newlist = EasyImageList()
            for img in self:
                newlist.append(img.resize(width, height, False))
            return newlist
            
    def save(self, folder = "./", root = "image"):
        n = len(self)
        digits = int(np.log10(n)) +1
        results = EasyImageFileList()
        for i in range(n):
            newpath = folder + "/" + root + str(i).zfill(digits) + ".jpg"
            self[i].save(newpath)
            results.append(EasyImageFile(newpath))
        return results
            
    def search(self, word):
        imgs = []
        for img in self:
            classes = img.classify() #TODO: Batch and classify multiple images together
            if word in classes.keys():
                imgs.append((img, classes[word]))
        return sorted(imgs, key=lambda x: x[1], reverse = True)
        
        
        

class EasyImageFileList(EasyImageList):
    
    def __init__(self, x = []):
        if isinstance(x, str):
            tmp = load_image_dir(x)
            [self.append(t) for t in tmp]
        else:
            super(EasyImageFileList, self).__init__(x)

    def __add__(self, x):
        if isinstance(x, EasyImageFileList):
            return EasyImageFileList(super(EasyImageFileList, self).__add__(x))
    
    def __iadd__(self, x):
        if isinstance(x, EasyImageFileList):
            return super(EasyImageFileList, self).__iadd__(x)
    
    def append(self, x):
        if isinstance(x, EasyImageFile):
            super(EasyImageFileList, self).append(x)
        else:
            try:
                tmp = EasyImageFile(x)
                super(EasyImageFileList, self).append(tmp)
            except NotAnImage:
                pass
    
    def remove_exif(self):
        [img.remove_exif() for img in self]

class EasyFaceList(EasyImageList):
    
    def __add__(self, x):
        if isinstance(x, EasyFaceList):
            return EasyFaceList(super(EasyFaceList, self).__add__(x))
    
    def __iadd__(self, x):
        if isinstance(x, EasyFaceList):
            return super(EasyFaceList, self).__iadd__(x)
    
    def append(self, x):
        if isinstance(x, EasyFace):
            super(EasyFaceList, self).append(x)
    
    def detect_faces(self, detector = None):
        return self
    
    def resize(self, width, height, inplace = False): #inplace is a dummy
        results = EasyImageList()
        for face in self:
            img = EasyImage(face.getimg())
            img.resize(width, height, inplace = True)
            results.append(img)
        return results

class ImageFileList(list):
    
    def __init__(self, x = []):
        for a in x:
            self.append(a)
    
    def __add__(self, x):
        if isinstance(x, str):
            return ImageFileList(super(ImageFileList, self).__add__(x))
    
    def __iadd__(self, x):
        if isinstance(x, str):
            return super(ImageFileList, self).__iadd__(x)
    
    def append(self, x):
        if isinstance(x, str):
            try:
                EasyImageFile(x)
                super(ImageFileList, self).append(x)
            except NotAnImage:
                pass
            
    def detect_faces(self, detector = None):
        """
Untested; need to test soon.
        """
        #TODO: Test this function
        faces = EasyFaceList()
        for i in self:
            tmp = EasyImageFile(i)
            faces += tmp.detect_faces()
        return faces
    
    def resize(self, height, width):
        result = EasyImageList()
        for f in self:
            tmp = EasyImage(f)
            tmp.resize(height, width)
            result.append(tmp)
        return result
            
            


def faces_in_dir(inputdir, detector = default_detector):
    mydir = path(inputdir)
    faces = []
    for f in mydir.files():
        try:
            ei = EasyImageFile(f)
            newfaces = ei.detect_faces(detector = default_detector)
            faces += newfaces
        except (NotAnImage, NotFace):
            pass
    return faces