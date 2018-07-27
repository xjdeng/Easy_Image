# Based loosely off of 
# http://blog.outcome.io/pytorch-quick-start-classifying-an-image/

import io, os, json
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable

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