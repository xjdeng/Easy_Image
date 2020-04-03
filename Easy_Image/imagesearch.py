import pandas as pd
import numpy as np
from path import Path as path
try:
    from . import detect
except ImportError:
    import detect
    
def run(start = "./", batch = 100):
    idxfile = "{}/image_index.csv".format(start)
    filequeue = list(path(start).walkfiles())
    try:
        existing = pd.read_csv(idxfile, index_col = 0)
        lookup = {f:s for (f,s) in zip(existing.index, existing['mtime'])}
    except IOError:
        lookup = {}
        columns = ['mtime'] + list(range(0,1440))
        existing = pd.DataFrame(columns=columns)
    for i,f in enumerate(filequeue):
        mtime = f.mtime
        fpath = str(f).replace("\\","/")
        print(fpath)
        if lookup.get(fpath) != mtime:
            #Index new file
            try:
                ei = detect.EasyImageFile(f)
                existing.loc[fpath] = [mtime] + ei.describe()
                print("Extracted")
            except detect.NotAnImage:
                print("Skipping due to error")
        else:
            #Skip existing file
            print("Skipping existing file")
        print("{} out of {} files completed".format(1+i, len(filequeue)))
        if i+1 % batch == 0:
            print("Saving results")
            existing.to_csv(idxfile)
    existing.to_csv(idxfile)

def search(img, start = "./", prefix = ""):
    if isinstance(img, str):
        img = detect.EasyImageFile(img)
    desc = np.array(img.describe())
    try:
        existing = pd.read_csv(start, index_col = 0)
    except IOError:
        existing = pd.read_csv("{}/image_index.csv".format(start), index_col = 0)
    dist = []
    index = []
    for i, row in existing.iterrows():
        if str(i).startswith(prefix):
            dist.append(np.linalg.norm(desc - np.array(row[1:])))
            index.append(i)
    output = pd.DataFrame(index = index)
    output['dist'] = dist
    output.sort_values('dist')
    print(output.head())
    return output