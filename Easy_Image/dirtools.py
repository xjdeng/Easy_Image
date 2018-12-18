import numpy as np
from path import Path as path
from PIL import Image
import random


def _isimage(myfile):
    try:
        Image.open(myfile)
        return True
    except IOError:
        return False
    
def _norm(mylist):
    s = sum(mylist)
    return [i/s for i in mylist]

def copy(mylist, mydest):
    [f.copy(mydest) for f in mylist]

def move(mylist, mydest):
    [f.move(mydest) for f in mylist]

def run(basedir, maxresults, scheme, minfiles = 10):
    dirfiles = {}
    for d in path(basedir).walkdirs():
        files = d.files()
        random.shuffle(files)
        if len(files) > minfiles:
            dirfiles[d] = files
    results = []
    keys = list(dirfiles.keys())
    goahead = False
    maxresults = min(maxresults, len(dirfiles))
    remaining = maxresults
    choose_n = min(len(keys), remaining)
    while goahead == False:
        # remaining = maxresults - len(results)
        # choose_n = min(len(keys), remaining)
        probs = _norm([scheme(d) for d in keys])
        choices = np.random.choice(keys, choose_n, False, probs)
        for c in choices:
            for f in dirfiles[c]:
                if _isimage(f):
                    results.append(f)
                    break
        remaining = maxresults - len(results)
        choose_n = min(len(keys), remaining)
        if choose_n <= 0:
            goahead = True
    return results

def run_eqwt(basedir, maxresults, minfiles = 10):
    return run(basedir, maxresults, lambda x:1, minfiles)

def run_newer_bias(basedir, maxresults, minfiles = 10):
    f = lambda x:1.0/np.log(x.ctime)
    return run(basedir, maxresults, f, minfiles)

def run_very_recent_bias(basedir, maxresults, minfiles = 10):
    return run(basedir, maxresults, lambda x:1.0/x.ctime, minfiles)

def run_smaller_bias(basedir, maxresults, minfiles = 10):
    return run(basedir, maxresults, lambda x:1.0/len(x.files()), minfiles)

def run_recent_bias_with_files(basedir, maxresults, minfiles = 10):
    f = lambda x:len(x.files())/np.log(x.ctime)
    return run(basedir, maxresults, f, minfiles)

def train_test_split(source, dest, valid_pct, inplace = False):
    p_source = path(source)
    dirs = p_source.dirs()
    dir_names = [str(a.name) for a in dirs]
    path(dest).mkdir_p()
    p_train = path(dest + "/train")
    p_valid = path(dest + "/valid")
    p_train.mkdir_p()
    p_valid.mkdir_p()
    for n in dir_names:
        path(p_train + "/" + n).mkdir_p()
        path(p_valid + "/" + n).mkdir_p()
    for d in dirs:
        files = d.files()
        random.shuffle(files)
        split = round(valid_pct * len(files))
        if inplace == False:
            operation = copy
        else:
            operation = move
        operation(files[0:split], p_valid + "/" + d.name)
        operation(files[split:], p_train + "/" + d.name)