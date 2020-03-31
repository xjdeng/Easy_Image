import torchvision.transforms as T
import torchvision
import cv2
from Easy_Image import detect, get_image_size as gis
import numpy as np
from path import Path as path
from rectpack import newPacker
from collections import defaultdict, deque
import random
import uuid
import gc


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

def get_prediction(ei, threshold=0.99):
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
    try:
        h,w = mask.shape
    except ValueError:
        return None
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

def mask_coords(ei, mask):
    try:
        h,w = mask.shape
    except ValueError:
        return None
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
    return x0,y0,x1,y1

def combine(x1, x2):
    lx1, ly1, rx1, ry1 = x1
    lx2, ly2, rx2, ry2 = x2
    if (lx1 > rx2 or lx2 > rx1):
        return None
    if (ly1 < ry2 or ly2 < ry1):
        return None
    lx = min(lx1, lx2)
    ly = min(ly1, ly2)
    rx = max(rx1, rx2)
    ry = max(ry1, ry2)
    return lx, ly, rx, ry

def join_overlapping(rects):
    if len(rects) == 1:
        return rects
    results = []
    while len(rects) > 0:
        test = rects.pop()
        combined = False
        for i,r in enumerate(rects):
            comb = combine(r, test)
            if comb:
                rects[i] = comb
                combined = True
                break
        if not combined:
            results.append(test)
    return results
            
def extract_masks(ei, outdir = None, obj = "person"):
    masks, _, pred_class = get_prediction(ei)
    indices = [i for i, x in enumerate(pred_class) if x == obj]
    h,w = ei.getimg().shape[0:2]
    mastermask = np.zeros((h,w))
    coords = []
    for i in indices:
        test = mask_coords(ei, masks[i])
        if test is not None:
            mastermask = np.logical_or(mastermask, masks[i])
            coords.append(test)
    if len(coords) > 1:
        coords = join_overlapping(coords)
    output = detect.EasyImageList()    
    for c in coords:
        x0,y0,x1,y1 = c
        h1 = x1 - x0
        w1 = y1 - y0
        mask2 = mastermask[x0:x1+1,y0:y1+1]
        newimg = ei.getimg()[x0:x1+1,y0:y1+1]
        gc.collect()
        for i in range(0,h1):
            for j in range(0, w1):
                if mask2[i,j] == 0:
                    newimg[i,j] = white
        test = detect.EasyImage(newimg)
        output.append(test)
        if outdir:
            newfile = path("{}/{}.png".format(outdir, str(uuid.uuid4())))
            while newfile.exists():
                newfile = path("{}/{}.png".format(outdir, str(uuid.uuid4())))
            test.save(newfile)
    return output
    

def generate_from_dir(maskdir, outdir, h, w, n = 1, max_h = 500, max_w = 500):
    """
    Important
    """
    files = path(maskdir).files()
    path(outdir).mkdir_p()
    rectangles = []
    for i,f in enumerate(files):
        try:
            w0,h0 = gis.get_image_size(f)
            if ((h0 > max_h) or (w0 > max_w)):
                if h0/max_h > w0/max_w:
                    new_h = max_h
                    new_w = int(round(new_h*w0/h0))
                else:
                    new_w = max_w
                    new_h = int(round(new_w*h0/w0))
                w0, h0 = new_w, new_h
            rectangles.append((h0, w0, i))
        except gis.UnknownImageFormat:
            pass
    random.shuffle(rectangles)
    bins = [(h,w)]*n
    packer = newPacker(rotation=False)
    for r in rectangles:
        packer.add_rect(*r)
    for b in bins:
        packer.add_bin(*b)
    packer.pack()
    all_rects = packer.rect_list()
    nbins = len(packer)
    bindict = defaultdict(list)
    for rect in all_rects:
        b, x0, y0, ww, hh, rid = rect
        x1 = x0 + ww
        y1 = y0 + hh
        bindict[b].append((x0,x1,y0,y1,rid))
    for i in range(0, nbins):
        img = 255*np.ones((h,w,3))
        for x0,x1,y0,y1,rid in bindict[i]:
            ei = detect.EasyImageFile(files[rid])
            h0 = x1 - x0
            w0 = y1 - y0
            if (h0,w0) != ei.getimg().shape[0:2]:
                ei.resize(w0, h0)
            img[x0:x1,y0:y1] = ei.getimg()
        cv2.imwrite("{}/collage_{}.jpg".format(outdir, i), img)

def extract_dir_masks(imgdir, outdir, obj = "person"):
    """
    Important
    """
    path(outdir).mkdir_p()
    for f in path(imgdir).files():
        gc.collect()
        try:
            print(f)
            ei = detect.EasyImageFile(f)
            extract_masks(ei, outdir, obj)
            print("Success!")
        except Exception as e:
            print(e)


def extract_obj(ei, obj="person"):
    masks, _, pred_class = get_prediction(ei)
    indices = [i for i, x in enumerate(pred_class) if x == obj]
    output = detect.EasyImageList()
    for i in indices:
        test = maskimg(ei, masks[i])
        if test is not None:
            output.append(test)
    return output



def generate_collages0(imgdir, outdir, h, w, n = 1, max_h = 500, max_w = 500, obj = "person"):
    path(outdir).mkdir_p()
    masks = detect.EasyImageList()
    for f in path(imgdir).files():
        try:
            newmasks = extract_obj(detect.EasyImageFile(f), obj)
            for m in newmasks:
                h0,w0 = m.getimg().shape[0:2]
                if ((h0 > max_h) or (w0 > max_w)):
                    if h0/max_h > w0/max_w:
                        new_h = max_h
                        new_w = int(round(new_h*w0/h0))
                    else:
                        new_w = max_w
                        new_h = int(round(new_w*h0/w0))
                    m.resize(new_w, new_h)
            masks += newmasks
        except detect.NotAnImage:
            pass
    rectangles = []
    for i,m in enumerate(masks):
        h0,w0 = m.getimg().shape[0:2]
        rectangles.append((h0,w0,i))
    random.shuffle(rectangles)
    bins = [(h,w)]*n
    packer = newPacker(rotation=False)
    for r in rectangles:
        packer.add_rect(*r)
    for b in bins:
        packer.add_bin(*b)
    packer.pack()
    all_rects = packer.rect_list()
    nbins = len(packer)
    bindict = defaultdict(list)
    for rect in all_rects:
        b, x0, y0, ww, hh, rid = rect
        x1 = x0 + ww
        y1 = y0 + hh
        bindict[b].append((x0,x1,y0,y1,rid))
    for i in range(0, nbins):
        img = 255*np.ones((h,w,3))
        for x0,x1,y0,y1,rid in bindict[i]:
            img[x0:x1,y0:y1] = masks[rid].getimg()
        cv2.imwrite("{}/collage_{}.jpg".format(outdir, i), img)
        