import pandas as pd
import numpy as np
from path import Path as path
try:
    from . import detect
except ImportError:
    import detect
import gc
import copy
import dlib
from tqdm import tqdm

def smartwalkfiles_old(start):
    dirs = path(start).dirs()
    gooddirs = []
    for d in dirs:
        if ("$RECYCLE.BIN" not in d) & ("System Volume Information" not in d) \
            &(".Spotlight-V100" not in d):
            gooddirs.append(d)
    files = path("./").files()
    for d in gooddirs:
        files += d.walkfiles()
    return files

def smartwalkfiles(start, exclusions = []):
    exclusions = set(exclusions)
    files = path(start).files()
    for d in path(start).dirs():
        if str(d) in exclusions:
            continue
        try:
            files += smartwalkfiles(d, exclusions)
        except PermissionError:
            pass
    return files

def same_mtime(t1, t2):
    if not t1:
        return False
    if t1 == t2:
        return True
    diff = abs(t1 - t2)
    if diff % 3600 == 0:
        return True
    if diff < 0.001:
        return True
    return False

def smart_lookup(mtimes, name, mtime):
    tmp = mtimes.get((name, mtime), None)
    if tmp:
        return tmp
    tmp = mtimes.get((name, mtime + 3600), None)
    if tmp:
        return tmp
    return mtimes.get((name, mtime - 3600), None)
    
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
            remove_filequeue = []
            for f in lookup.keys():
                if f not in filequeue:
                    pathf = path(f)
                    for f2 in filequeue:
                        fp = path(f2)
                        if (same_mtime(fp.mtime, lookup[f])) & (pathf.name == fp.name):
                            print("Moved file detected: {}".format(f))
                            addition.loc[f2] = existing.loc[f]
                            remove_filequeue.append(f2)
                    remove_keys.append(f)
                    print("{} no longer found, deleting from database".format(f))
            existing.drop(remove_keys,0,inplace=True)
            for f in remove_keys:
                del lookup[f]
            for f2 in remove_filequeue:
                try:
                    filequeue.remove(f2)
                except KeyError:
                    print("Error: key {} not found".format(f2))
        except IOError:
            lookup = {}
            existing = pd.DataFrame(columns=columns)
        j = 0
        for i,f0 in enumerate(filequeue):
            f = path(f0)
            mtime = f.mtime
            fpath = str(f).replace("\\","/")
            print(fpath)
            #if lookup.get(fpath) != mtime:
            test = lookup.get(fpath)
            if (test is not None) & ((test - mtime) % 3600 == 0):
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
    #M = existing.loc[index].to_numpy()[:,1:]
    cols = [str(i) for i in range(0,1440)]
    dist = np.linalg.norm(desc - existing.loc[index][cols], axis=1)
    output = pd.DataFrame(index = index)
    output['dist'] = dist
    output.sort_values('dist', inplace=True)
    print(output.head())
    return output

def load_faces(start = "./", prefix = ""):
    try:
        existing = pd.read_csv(start, index_col = 0, low_memory = True)
    except IOError:
        existing = pd.read_csv("{}/face_index.zip".format(start), index_col = 0, low_memory = True)
    existing = existing[existing['faces'] > 0]
    if len(prefix) > 0:
        existing = existing[existing['file'].str.startswith(prefix)]
    return existing

def search_faces(encoding, start = "./", prefix = ""):
    existing = load_faces(start, prefix)
    df = pd.DataFrame()
    df['file'] = existing['file']
    cols = [str(i) for i in range(128)]
    g = np.linalg.norm(encoding - existing[cols], axis=1)
    df['dist'] = g
    df.sort_values('dist', inplace=True)
    print(df.head())
    return df

def face_vector(f, fpath, mtime):
    try:
        ei = detect.EasyImageFile(f)
        faces = ei.detect_faces_simple()
        nfaces = len(faces)
        newrow = [fpath, mtime, nfaces]
        print("Found {} faces".format(nfaces))
        if nfaces == 0:
            newrow += list(range(0,132))
            return [newrow]
        else:
            results = []
            for face in faces:
                newrow2 = copy.deepcopy(newrow)
                newrow2 += [face.face.left(), face.face.top(), face.face.right(), face.face.bottom()]
                newrow2 += list(face.face_encoding())
                results.append(newrow2)
            return results
    except detect.NotAnImage:
        print("Skipping due to error")
        return [[fpath, mtime] + list(range(0,133))]
    

def run_meta(func, columns, default_file, start = "./", batch = 1000):
    idxfile = "{}/{}".format(start, default_file)
    #addition = pd.DataFrame(columns=columns)
    add_idx, add = [],[]
    existing = None
    def save(ex, ad):
        addd_idx, addd = ad
        ad = pd.DataFrame(addd, columns=columns, index=addd_idx)
        #ad = ad.astype(dtypes)
        print("Saving results")
        gc.collect()
        ex = ex.append(ad, sort=True)
        gc.collect()
        oldpaths = list(ex['file'])
        ex['file'] = [p.replace(start, "./") for p in ex['file']]
        ex[columns].to_csv(idxfile)
        ex['file'] = oldpaths
    try:
        filequeue = set([str(f).replace("\\","/") for f in smartwalkfiles(start)])
        try:
            existing = pd.read_csv(idxfile, index_col = 0, low_memory=True)#, dtype=dtypes)
            existing.index = list(range(0, len(existing)))
            existing.columns = columns
            existing['file'] = [p.replace("./",start) for p in existing['file']]
            if len(existing.index) == 0:
                idx = 0
            else:
                idx = 1 + max(existing.index)
            gc.collect()
            lookup = {f.replace("./",start):s for (f,s) in zip(existing['file'], existing['mtime'])}
            remove_keys = []
            remove_filequeue = []
            mtimes = None
            for f in lookup.keys():
                if f not in filequeue:
                    if not mtimes:
                        
                        mtimes = {}
                        for f2 in filequeue:
                            fp = path(f2)
                            mtimes[(fp.name, fp.mtime)] = f2
                    pathf = path(f)
                    #f2 = mtimes.get((pathf.name, lookup[f]), None)
                    f2 = smart_lookup(mtimes, pathf.name, lookup[f])
                    if f2 is not None:
                        print("Moved file detected: {}".format(f))
                        tmp = existing[existing['file']==f]
                        for index in tmp.index:
                            newrow = list(tmp.loc[index])
                            newrow[0] = f2
                            #addition.loc[idx] = newrow
                            add_idx.append(idx)
                            add.append(newrow)
                            idx += 1
                        remove_filequeue.append(f2)
                    remove_keys.append(f)
                    print("{} no longer found, deleting from database".format(f))
            remove_keys = set(remove_keys)
            remove_idx = []
            print("Stage1")
            for idx2 in existing.index:
                f = existing['file'].loc[idx2]
                try:
                    if f in remove_keys:
                        print("Looking at file: " + f)
                        remove_idx.append(idx2)
                        print("Deleting")
                        try:
                            print("Looking up")
                            del lookup[f]
                        except KeyError:
                            print("KeyError")
                except TypeError:
                    remove_idx.append(idx)
                    print("TypeError")
            print("Stage2")
            existing.drop(existing.index[remove_idx], inplace=True)
            print("Stage3")
            for f2 in remove_filequeue:
                try:
                    filequeue.remove(f2)
                except KeyError:
                    print("File {} not found, skipping".format(f2))
        except IOError:
            lookup = {}
            existing = pd.DataFrame(columns=columns)
            #existing = existing.astype(dtypes)
            idx = 0
        j = 0
        print(len(filequeue))
        existing_files = set(existing['file'])
        for i,f0 in enumerate(tqdm(filequeue)):
            f = path(f0)
            mtime = f.mtime
            fpath = str(f).replace("\\","/")
            #print(fpath)
            if not same_mtime(lookup.get(fpath), mtime):#lookup.get(fpath) != mtime:
                #print("Adding: {}".format(fpath))
                #print(lookup.get(fpath), mtime)
                j += 1
                #print(fpath)
                #print(existing.tail())
                if fpath in existing_files:
                    existing.drop(existing.loc[existing['file'] == fpath].index, inplace=True)
                #print(existing.tail())
                #Begin snippet
                try:
                    additions = func(f, fpath, mtime)
                    for ad in additions:
                        #addition.loc[idx] = add
                        add_idx.append(idx)
                        add.append(ad)
                        idx += 1
                except PermissionError:                    
                    pass#print("Permission Error, skipping")
                #End snippet
            else:
    
                pass#print("Skipping existing file")
            #print("{} out of {} files completed".format(1+i, len(filequeue)))
            if (j+1) % batch == 0:
                #print("Appending current batch")
                gc.collect()
                addition = pd.DataFrame(add, columns=columns, index=add_idx)
                #addition = addition.astype(dtypes)
                existing = existing.append(addition, sort=True)
                existing_files = set(existing['file'])
                gc.collect()
                #print("Saving results")
                existing[columns].to_csv(idxfile)
                gc.collect()
                #addition = pd.DataFrame(columns=columns, index=add_idx)
                add_idx, add = [], []
                gc.collect()
                j += 1            
        save(existing, (add_idx, add))
    except KeyboardInterrupt as e:
        print(e)
        save(existing, (add_idx, add))
    #except Exception as e:
    #    print(e)
    #    print("Outer Exception")
    #    save(existing, (add_idx, add))
        
def run_faces(start = "./", fname = "face_index.csv", batch = 1000, exclusions = []):
    columns = ['file','mtime','faces','left','top','right','bottom'] + list(range(0,128))
    run_meta(face_vector, columns, fname, start, batch, exclusions)

def run_faces_old(start = "./", batch = 1000, faceimgs = False):
    if faceimgs:
        idxfile = "{}/faceimgs_index.zip".format(start)
    else:
        idxfile = "{}/face_index.zip".format(start)
    columns = ['file','mtime','faces','left','top','right','bottom'] + list(range(0,128))
    addition = pd.DataFrame(columns=columns)
    existing = None
    def save(ex, ad):
        print("Saving results")
        gc.collect()
        ex = ex.append(ad, sort=True)
        gc.collect()
        ex[columns].to_csv(idxfile)
    try:
        filequeue = set([str(f).replace("\\","/") for f in smartwalkfiles(start)])
        try:
            existing = pd.read_csv(idxfile, index_col = 0, low_memory=True)
            existing.index = list(range(0, len(existing)))
            existing.columns = columns
            if len(existing.index) == 0:
                idx = 0
            else:
                idx = 1 + max(existing.index)
            gc.collect()
            lookup = {f:s for (f,s) in zip(existing['file'], existing['mtime'])}
            remove_keys = []
            remove_filequeue = []
            mtimes = None
            for f in lookup.keys():
                if f not in filequeue:
                    if not mtimes:
                        
                        mtimes = {}
                        for f2 in filequeue:
                            fp = path(f2)
                            mtimes[(fp.name, fp.mtime)] = f2
                    pathf = path(f)
                    f2 = mtimes.get((pathf.name, lookup[f]), None)
                    if f2 is not None:
                        print("Moved file detected: {}".format(f))
                        tmp = existing[existing['file']==f]
                        for index in tmp.index:
                            newrow = list(tmp.loc[index])
                            newrow[0] = f2
                            addition.loc[idx] = newrow
                            idx += 1
                        remove_filequeue.append(f2)
                    remove_keys.append(f)
                    print("{} no longer found, deleting from database".format(f))
            remove_keys = set(remove_keys)
            remove_idx = []
            print("Stage1")
            for idx2 in existing.index:
                f = existing['file'].loc[idx2]
                try:
                    if f in remove_keys:
                        print("Looking at file: " + f)
                        remove_idx.append(idx2)
                        print("Deleting")
                        try:
                            print("Looking up")
                            del lookup[f]
                        except KeyError:
                            print("KeyError")
                except TypeError:
                    remove_idx.append(idx)
                    print("TypeError")
            print("Stage2")
            existing.drop(existing.index[remove_idx], inplace=True)
            print("Stage3")
            for f2 in remove_filequeue:
                filequeue.remove(f2)
        except IOError:
            lookup = {}
            existing = pd.DataFrame(columns=columns)
            idx = 0
        j = 0
        print(len(filequeue))
        for i,f0 in enumerate(filequeue):
            f = path(f0)
            mtime = f.mtime
            fpath = str(f).replace("\\","/")
            print(fpath)
            if lookup.get(fpath) != mtime:
                j += 1
                existing.drop(existing.loc[existing['file'] == fpath].index, inplace=True)
                #Begin snippet
                try:
                    ei = detect.EasyImageFile(f)
                    if faceimgs:
                        h,w = ei.getimg().shape[0:2]
                        rect = dlib.rectangle(0,0,w,h)
                        ef = detect.EasyFace(ei, rect)
                        faces = detect.EasyFaceList([ef])
                    else:
                        faces = ei.detect_faces_simple()
                    nfaces = len(faces)
                    newrow = [fpath, mtime, nfaces]
                    print("Found {} faces".format(nfaces))
                    if nfaces == 0:
                        newrow += list(range(0,132))
                        addition.loc[idx] = newrow
                        idx += 1
                    else:
                        for face in faces:
                            newrow2 = copy.deepcopy(newrow)
                            newrow2 += [face.face.left(), face.face.top(), face.face.right(), face.face.bottom()]
                            newrow2 += list(face.face_encoding())
                            addition.loc[idx] = newrow2
                            idx += 1
                except detect.NotAnImage:
                    addition.loc[idx] = [fpath, mtime] + list(range(0,133))
                    print("Skipping due to error")
                except PermissionError:                    
                    print("Permission Error, skipping")
                #End snippet
            else:
                print("Skipping existing file")
            print("{} out of {} files completed".format(1+i, len(filequeue)))
            if (j+1) % batch == 0:
                print("Appending current batch")
                gc.collect()
                existing = existing.append(addition, sort=True)
                gc.collect()
                print("Saving results")
                existing[columns].to_csv(idxfile)
                gc.collect()
                addition = pd.DataFrame(columns=columns)
                gc.collect()
                j += 1            
        save(existing, addition)
    except Exception as e:
        print(e)
        print("Outer Exception")
        save(existing, addition)