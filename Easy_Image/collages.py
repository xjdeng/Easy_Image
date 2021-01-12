import copy
import cv2
from Easy_Image import detect
import numpy as np
from path import Path as path

def _combine(list_of_images, height = None):
    first = list_of_images[0]
    if height is None:
        height = first.getimg().shape[0]
    width = 0
    for img in list_of_images:
        h,w,_ = img.getimg().shape
        if h != height:
            img.resize(round(1.0*height*w/h), height)
        width += img.getimg().shape[1]
        if np.random.random() > 0.5:
            img._img = cv2.flip(img.getimg(), 1) #0.5 prob of flipping
    newimg = np.zeros((height, width, 3))
    start = 0
    end = 0
    for img in list_of_images:
        end += img.getimg().shape[1]
        newimg[:,start:end,:] = img.getimg()
        start = end
    return detect.EasyImage(newimg)
        
    

def make_horizontal(list_of_lists, min_w, tgt_h, num_images, scheme = 'eqwt',\
                    save= False):
    """
    list_of_lists: list of EasyImageLists to pick images from or a directory of directories
    tgt_w: target width (may NOT be exact)
    tgt_h: target height (will be EXACT)
    num_images: number of images to generate
    scheme: string or function indicating the scheme for selecting images
    """
    if isinstance(list_of_lists, str):
        p = path(list_of_lists)
        #list_of_lists = [detect.load_image_dir(q) for q in p.dirs()]
        list_of_lists = [path(q).files() for q in p.dirs()]
    if scheme == "eqwt":
        scheme = scheme_eqwt
    elif scheme == "weighted":
        scheme = scheme_weighted
    images = detect.EasyImageList()
    for i in range(num_images):
        goahead = False
        width = 0
        list2 = copy.copy(list_of_lists)
        candidates = []
        while goahead == False:
            if (width >= min_w) or len(list2) == 0:
                goahead = True
            else:
                mydir = scheme(list2)
                if len(mydir) != 0:
                    choice0 = np.random.choice(mydir)
                    try:
                        choice = detect.EasyImageFile(choice0)
                    except detect.NotAnImage:
                        continue
                    h,w,_ = choice.getimg().shape
                    width += round(1.0*tgt_h*w/h)
                    candidates.append(choice)
        result = _combine(candidates, tgt_h)
        images.append(result)
    if save != False:
        if save == True:
            save = "./"
        images.save(save)
    return images
            
            

def scheme_eqwt(list_of_lists):
    choice = np.random.randint(len(list_of_lists))
    return list_of_lists.pop(choice)

def scheme_weighted(list_of_lists):
    raw = [len(l) for l in list_of_lists]
    probs = [1.0*len(l)/sum(raw) for l in list_of_lists]
    choice = np.random.choice(range(0, len(list_of_lists)), p=probs)
    return list_of_lists.pop(choice)