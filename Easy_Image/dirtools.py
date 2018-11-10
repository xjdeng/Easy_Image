import numpy as np
from path import Path as path
from PIL import Image

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

def run(basedir, maxresults, scheme, minfiles = 10):
    dirfiles = {}
    for d in path(basedir).walkdirs():
        files = d.files()
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