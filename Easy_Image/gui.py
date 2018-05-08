from tkinter import filedialog

import pandas as pd
import cv2

try:
    from . import detect
except ImportError:
    import detect
    
up = 38
down = 40
left = 37
right = 39

def label_images(img_list, attribute_lists):
    flag = cv2.WINDOW_NORMAL
    images = len(img_list)
    attributes = len(attribute_lists)
    labels = [[] for i in range(images)]
    i = 0
    goahead = False
    while goahead == False:
        img = img_list[i].getimg()
        wname = str(hash(img.tostring()))
        cv2.namedWindow(wname, flag) 
        cv2.imshow(wname, img)
        print("Image {} of {}".format(i+1, images))
        goahead2 = False
        j = 0
        while goahead2 == False:
            print("On Attribute {}\n".format(j+1))
            print("Choices:\n")
            choices = len(attribute_lists[j])
            for k in range(0, min(9, choices)):
                print("{}. {}".format(k+1, attribute_lists[j][k]))
            print("\n")
            print("Left Arrow: Previous Attribute")
            if j + 1 == attributes:
                if i + 1 == images:
                    print("Right Arrow: FINISH LABELING")
                else:
                    print("Right Arrow: Next IMAGE")
            else:
                print("Right Arrow: Next Attribute")
            print("Up Arrow: Next Image")
            print("Down Arrow: Previous Image")
            key = cv2.waitKey()
            if key == up:
                if len(labels[i]) < attributes:
                    print("Finish labeling everything before moving on!")
                else:
                    if i + 1 == images:
                        goahead = True
                    i += 1
                    goahead2 = True
            elif key == down:
                if i == 0:
                    print("Already at the beginning.")
                else:
                    i -= 1
                    goahead2 = True
            elif key == left:
                if j == 0:
                    print("Already on the first attribute.")
                else:
                    j -= 1
            elif key == right:
                if j + 1 == attributes:
                    goahead2 = True
                    if i + 1 == images:
                        goahead = True
                    else:
                        i += 1
                elif j + 1 >= len(labels):
                    print("Label this one first!")
                else:
                    j += 1
                    goahead2 = True
            elif 49 <= key <= 57:
                idx = key - 49
                if idx >= choices:
                    print("Invalid Choice")
                else:
                    try:
                        labels[i][j] = attribute_lists[j][idx]
                    except IndexError:
                        labels[i].append(attribute_lists[j][idx])
                    j += 1
                    if len(labels[i]) == attributes:
                        goahead2 = True
                        i += 1
        cv2.destroyWindow(wname)
    result = pd.DataFrame()
    result.index = img_list
    for i in range(attributes):
        result[i] = [l[i] for l in labels]
    return result
            
        
    
    

def load():
    imgfile = filedialog.askopenfilename(initialdir = "/",\
                                         title = "Select Image:")
    return detect.EasyImageFile(imgfile)

def load_dir(recursive = False):
    imgdir = filedialog.askdirectory(initialdir = "/",\
                                         title = "Select Directory:")
    return detect.load_image_dir(imgdir, recursive = recursive)