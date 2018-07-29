#Taken from https://github.com/cetinsamet/age-estimation
import sys
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

try:
    from train import twoHiddenNet
    from img_to_vec import Img2Vec
except ImportError:
    from Easy_Image.train import twoHiddenNet
    from Easy_Image.img_to_vec import Img2Vec

import os

curdir = os.path.dirname(__file__)

if len(curdir) == 0:
    MODEL_PATH = "misc/age_est_model.pt"
else:
    MODEL_PATH = os.path.dirname(__file__) + "/misc/age_est_model.pt"

model = twoHiddenNet()
model.load_state_dict(torch.load(MODEL_PATH))

def cv2_to_PIL(cv2_im):
    tmp = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(tmp)
    return pil_im


def get_age_from_cv2(img):
    fe = Img2Vec(cuda=False)
    img2 = copy.copy(img)
    img3 = cv2_to_PIL(img2)
    img3 = img3.resize((224,224))
    image_feats = fe.get_vec(img3).reshape(1, -1)
    estimated_age = model(Variable(torch.from_numpy(image_feats).float()))
    return estimated_age.data.cpu().numpy()[0][0]
    

def get_feats(image_path):
    fe = Img2Vec(cuda=False)  # change this if you use Cuda version of the PyTorch.
    img = Image.open(image_path)
    img = img.resize((224, 224))
    feats = fe.get_vec(img).reshape(1, -1)
    return feats

def main(argv):
    if len(argv) != 1:
        print("Usage: python3 age_est.py imagepath")
        exit()
    


    image_path = argv[0]
    image_feats = get_feats(image_path)

    estimated_age = model(Variable(torch.from_numpy(image_feats).float()))
    print("Estimated Age : %.1f" % estimated_age.data.cpu().numpy()[0][0])
    return

if __name__ == '__main__':
    main(sys.argv[1:])