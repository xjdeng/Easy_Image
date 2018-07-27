# Based loosely off of 
# http://blog.outcome.io/pytorch-quick-start-classifying-an-image/

import os, json
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import cv2

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

def cv2_to_PIL(cv2_im):
    tmp = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(tmp)
    return pil_im

def PIL_to_raw(img_pil):
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    fc_out = squeeze(img_variable)
    return fc_out.data.numpy()

