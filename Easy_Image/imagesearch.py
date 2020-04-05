import pandas as pd
import numpy as np
from path import Path as path
try:
    from . import detect
except ImportError:
    import detect
import gc

def smartwalkfiles(start):
    dirs = path(start).dirs()
    gooddirs = []
    for d in dirs:
        if ("$RECYCLE.BIN" not in d) & ("System Volume Information" not in d):
            gooddirs.append(d)
    files = []
    for d in gooddirs:
        files += d.walkfiles()
    return files
    
def run(start = "./", batch = 5000):
    idxfile = "{}/image_index.zip".format(start)
    columns = ['mtime'] + list(range(0,1440))
    addition = pd.DataFrame(columns=columns)
    existing = None
    def save(ex, ad):
        print("Saving results")
        gc.collect()
        ex = ex.append(ad, sort=True)
        gc.collect()
        ex.to_csv(idxfile)
    try:
        #filequeue = set([str(f).replace("\\","/") for f in path(start).walkfiles()])
        filequeue = set([str(f).replace("\\","/") for f in smartwalkfiles(start)])
        try:
            existing = pd.read_csv(idxfile, index_col = 0, low_memory=True)
            existing.columns = columns
            gc.collect()
            lookup = {f:s for (f,s) in zip(existing.index, existing['mtime'])}
            remove_keys = []
            for f in lookup.keys():
                if f not in filequeue:
                    remove_keys.append(f)
                    print("{} no longer found, deleting from database".format(f))
            existing.drop(remove_keys,0,inplace=True)
            for f in remove_keys:
                del lookup[f]
        except IOError:
            lookup = {}
            existing = pd.DataFrame(columns=columns)
        j = 0
        for i,f0 in enumerate(filequeue):
            f = path(f0)
            mtime = f.mtime
            fpath = str(f).replace("\\","/")
            print(fpath)
            if lookup.get(fpath) != mtime:
                #Index new file
                j += 1
                try:
                    ei = detect.EasyImageFile(f)
                    addition.loc[fpath] = [mtime] + ei.describe()
                    print("Extracted")
                except detect.NotAnImage:
                    addition.loc[fpath] = [mtime] + [0]*1440
                    print("Skipping due to error")
            else:
                #Skip existing file
                print("Skipping existing file")
            print("{} out of {} files completed".format(1+i, len(filequeue)))
            if (j+1) % batch == 0:
                print("Appending current batch")
                gc.collect()
                existing = existing.append(addition, sort=True)
                gc.collect()
                print("Saving results")
                existing.to_csv(idxfile)
                gc.collect()
                addition = pd.DataFrame(columns=columns)
                gc.collect()
                j += 1
        save(existing, addition)
    except Exception as e:
        print(e)
        save(existing, addition)

def search(img, start = "./", prefix = ""):
    if isinstance(img, str):
        img = detect.EasyImageFile(img)
    desc = np.array(img.describe())
    try:
        existing = pd.read_csv(start, index_col = 0, low_memory = True)
    except IOError:
        existing = pd.read_csv("{}/image_index.zip".format(start), index_col = 0, low_memory = True)
    index = [i for (i,_) in existing.iterrows() if str(i).startswith(prefix)]
    M = existing.loc[index].to_numpy()[:,1:]
    dist = np.linalg.norm(desc - M, axis=1)
    output = pd.DataFrame(index = index)
    output['dist'] = dist
    output.sort_values('dist', inplace=True)
    print(output.head())
    return output