#Based off of: https://github.com/yu4u/age-gender-estimation
import cv2
import numpy as np
from wide_resnet import WideResNet
from keras.backend import backend
from keras.utils.data_utils import get_file

#https://superuser.com/questions/470664/how-to-download-dropbox-files-using-wget-command
tensorflow_model = "https://www.dropbox.com/s/ukm1dxca0vve6gn/weights.18-4.06.hdf5?dl=1"
tensorflow_file = "weights.18-4.06.hdf5"
theano_model = "https://www.dropbox.com/s/e3q3urdwv4ng2sz/weights_theano.16-3.99.hdf5?dl=1"
theano_file = "weights_theano.16-3.99.hdf5"

if backend() == "tensorflow":
    modurl = tensorflow_model
    modfile = tensorflow_file
    modhash = '89f56a39a78454e96379348bddd78c0d'
    default_mod = get_file(modfile, modurl, cache_subdir="models", file_hash=modhash)
elif backend() == "theano":
    modurl = theano_model
    modfile = theano_file
    modhash = 'd632812726dc28d30db8b1f1c99f20a4'
    default_mod = get_file(modfile, modurl, cache_subdir="models", file_hash=modhash)
else:
    default_mod = None


def run(img, mod = default_mod, img_size = 64, depth = 16, k = 8):
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(mod)
    faces = np.empty((1, img_size, img_size, 3))
    faces[0,:,:,:] = cv2.resize(img, (img_size, img_size))
    results = model.predict(faces)
    predicted_genders = results[0][0][0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()[0]
    return (predicted_genders, predicted_ages)

#TODO: Document the run() function and make another version
#for lists of faces