# Based off of the sample code at:
# https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
import numpy as np
import cv2
from PIL import Image as pil_image
import gc

MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50
}

_PIL_INTERPOLATION_METHODS = {
    'nearest': pil_image.NEAREST,
    'bilinear': pil_image.BILINEAR,
    'bicubic': pil_image.BICUBIC,
}

def convert_opencv(cv2_im, grayscale=False, target_size=None,
             interpolation='nearest'):
    
#https://stackoverflow.com/questions/13576161/convert-opencv-image-into-pil-image-in-python-for-use-with-zbar-library
    
    cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    img = pil_image.fromarray(cv2_im)
#https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img

def classify(img, mod = 'inception'):
    K.clear_session()
    gc.collect()
    if mod in ("inception", "xception"):
        inputShape = (299, 299)
        preprocess = preprocess_input
    else:
        inputShape = (224, 224)
        preprocess = imagenet_utils.preprocess_input      
    if isinstance(img, str):
        img1 = load_img(img, target_size=inputShape)
    else:
        img1 = convert_opencv(img, target_size=inputShape)
    img2 = img_to_array(img1)
    img3 = np.expand_dims(img2, axis = 0)
    img4 = preprocess(img3)
    Network = MODELS[mod]
    model = Network(weights="imagenet")
    preds = model.predict(img4)
    return imagenet_utils.decode_predictions(preds)[0]

# TODO 1. Support Internet img inputs in classify()
# TODO 2. Support classifying multiple images at once.