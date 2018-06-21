#WARNING: Under Construction
import numpy as np

def make_horizontal(list_of_lists, tgt_w, tgt_h, num_images, scheme):
    """
    list_of_lists: list of EasyImageLists to pick images from
    tgt_w: target width (may NOT be exact)
    tgt_h: target height (will be EXACT)
    num_images: number of images to generate
    scheme: string or function indicating the scheme for selecting images
    """
    pass

def scheme_eqwt(list_of_lists):
    choice = np.random.randint(len(list_of_lists))
    return list_of_lists.pop(choice)

def scheme_weighted(list_of_lists):
    raw = [len(l) for l in list_of_lists]
    probs = [1.0*len(l)/sum(raw) for l in list_of_lists]
    choice = np.random.choice(range(0, len(list_of_lists)), p=probs)
    return list_of_lists.pop(choice)