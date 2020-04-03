import pandas as pd
import numpy as np
from path import Path as path
try:
    from . import detect
except ImportError:
    import detect
    
def run(start = "./", batch = 5000):
    try:
        idxfile = "{}/image_index.zip".format(start)
        filequeue = set([str(f).replace("\\","/") for f in path(start).walkfiles()])
        try:
            existing = pd.read_csv(idxfile, index_col = 0)
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
            columns = ['mtime'] + list(range(0,1440))
            existing = pd.DataFrame(columns=columns)
        j = 0
        for i,f0 in enumerate(filequeue):
            f = path(f0)
            mtime = f.mtime
            fpath = str(f).replace("\\","/")
            print(fpath)
            if lookup.get(fpath) != mtime:
                #Index new file
                try:
                    ei = detect.EasyImageFile(f)
                    existing.loc[fpath] = [mtime] + ei.describe()
                    j += 1
                    print("Extracted")
                except detect.NotAnImage:
                    existing.loc[fpath] = [mtime] + [0]*1440
                    print("Skipping due to error")
            else:
                #Skip existing file
                print("Skipping existing file")
            print("{} out of {} files completed".format(1+i, len(filequeue)))
            if (j+1) % batch == 0:
                print("Saving results")
                existing.to_csv(idxfile)
        existing.to_csv(idxfile)
    except KeyboardInterrupt:
        existing.to_csv(idxfile)

def search(img, start = "./", prefix = ""):
    if isinstance(img, str):
        img = detect.EasyImageFile(img)
    desc = np.array(img.describe())
    try:
        existing = pd.read_csv(start, index_col = 0)
    except IOError:
        existing = pd.read_csv("{}/image_index.zip".format(start), index_col = 0)
    index = [i for (i,_) in existing.iterrows() if str(i).startswith(prefix)]
    M = existing.loc[index].to_numpy()[:,1:]
    dist = np.linalg.norm(desc - M, axis=1)
    output = pd.DataFrame(index = index)
    output['dist'] = dist
    output.sort_values('dist', inplace=True)
    print(output.head())
    return output