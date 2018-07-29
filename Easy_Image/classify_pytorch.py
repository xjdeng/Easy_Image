# Based loosely off of 
# http://blog.outcome.io/pytorch-quick-start-classifying-an-image/

import os, json
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import numpy as np

squeeze = models.squeezenet1_1(pretrained=True)

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

squeeze = models.squeezenet1_1(pretrained=True)

raw_label_f = open(os.path.dirname(__file__) + "/misc/labels.json")
raw_labels = json.load(raw_label_f)
raw_label_f.close()

labels = {int(key):value for (key, value) in raw_labels.items()}

def best_prediction(raw):
    return labels[raw.argmax()]

def classify(cv2_img):
    img_pil = cv2_to_PIL(cv2_img)
    raw = PIL_to_raw(img_pil)
    preds = raw[0]
    return decode_predictions(preds)

def classify_multiple(cv2_list):
    return [classify(c) for c in cv2_list]

def combine_classifications(classified_list):
    tot = 0
    result = {}
    for c in classified_list:
        for d in c.keys():
            tot += np.float64(c[d])
    for c in classified_list:
        for d in c.keys():
            try:
                result[d] += np.float64(c[d])/tot
            except KeyError:
                result[d] = np.float64(c[d])/tot
    return result    

def cv2_to_PIL(cv2_im):
    tmp = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(tmp)
    return pil_im

def decode_predictions(preds, top = 5):
    top_indices = preds.argsort()[-top:][::-1]
    total = sum(preds[top_indices])
    results = {}
    for t in top_indices:
        pred = labels[t]
        results[pred] = preds[t]/total
    return results

def PIL_to_raw(img_pil):
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    fc_out = squeeze(img_variable)
    return fc_out.data.numpy()

