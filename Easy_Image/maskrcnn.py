import torchvision.transforms as T
import torchvision
import cv2
from Easy_Image import detect
import numpy as np

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
white = np.array([255,255,255])

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(ei, threshold=0.8):
    img = cv2.cvtColor(ei.getimg(),cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    try:
        pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    except IndexError:
        return [],[],[]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class

def maskimg(ei, mask):
    h,w = mask.shape
    x0,y0,x1,y1 = 0,0,h,w
    for i in range(0, h):
        if any(mask[i,:]):
            x0 = i
            break
    for i in range(h-1,-1,-1):
        if any(mask[i,:]):
            x1 = i
            break
    for i in range(0, w):
        if any(mask[:,i]):
            y0 = i
            break
    for i in range(w-1,-1,-1):
        if any(mask[:,i]):
            y1 = i
            break
    h1 = x1 - x0
    w1 = y1 - y0
    mask2 = mask[x0:x1+1,y0:y1+1]
    newimg = ei.getimg()[x0:x1+1,y0:y1+1]
    for i in range(0,h1):
        for j in range(0, w1):
            if mask2[i,j] == 0:
                newimg[i,j] = white
    return detect.EasyImage(newimg)

def extract_obj(ei, obj="person"):
    masks, _, pred_class = get_prediction(ei)
    indices = [i for i, x in enumerate(pred_class) if x == obj]
    return detect.EasyImageList([maskimg(ei, masks[i]) for i in indices])