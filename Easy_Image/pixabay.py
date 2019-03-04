from path import Path as path
import pixabay as python_pixabay
import tempfile
from sklearn.externals import joblib
import os
import time
import warnings
import json
import requests
import numpy as np
import re

cache_filename = "pixabay_cache.pkl"

api_key = None
cache_update_interval = 3600
cache_expiry = 24*3600
cache = {}

def cache_path():
    return tempfile.gettempdir() + "/" + cache_filename

def create_cache():
    ccache = {}
    ccache['last_saved'] = time.time()
    joblib.dump(ccache, cache_path())
    return ccache

try:
    cache = joblib.load(cache_path())
    if isinstance(cache, dict) == False:
        cache = create_cache()
except IOError:
    cache = create_cache()




def update_api_key():
    global api_key
    try:
        if api_key is None:
            api_key = os.environ['PIXABAY_API_KEY']
    except KeyError:
        message = """Pixabay API key not found.  Please get your key at 
https://pixabay.com/api/docs/ and either set it in your OS's PIXABAY_API_KEY 
environment variable or set it by calling the set_key() function.""".replace("\n","")
        warnings.warn(message)

update_api_key()

def cdn_to_larger(url):
    newname = re.sub("__[0-9]+","_{}".format("960_720"), url)
    return newname

def download(url, folder = "./"):
    try:
        filename = str(path(url).name)
        res = requests.get(url)
        res.raise_for_status()
        f = open(folder + "/" + filename, 'wb')
        for chunk in res.iter_content(100000):
            f.write(chunk)
    except Exception:
        pass
        
def download_google_file(google_file, folder = "./"):
    """
Do a Google image search limited to pixabay.com and get the download file
using these instructions:
https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb
Then use this script to grab the higher res photos.
    """
    f = open(google_file,'r')
    urls = f.read().split("\n")
    f.close()
    [download(cdn_to_larger(url), folder) for url in urls]

def download_query(myquery, destination = "./", imgtype = "largeImageURL"):
    imglist = images_from_query(myquery, imgtype)
    [download(url, destination) for url in imglist]

def images_from_query(myquery, imgtype = "largeImageURL"):
    if isinstance(myquery, list):
        results = []
        for item in myquery:
            results += images_from_query(item, imgtype)
        return list(set(results))
    return [m[imgtype] for m in  myquery['hits']]

def query(*args, **kwargs):
    try:
        return cache[(args, json.dumps(kwargs))][0]
    except KeyError:
        update_api_key()
        pix = python_pixabay.Image(api_key)
        results = pix.search(*args, **kwargs)
        update_cache((args, json.dumps(kwargs)), results)
        return results

def query_all_pages(*args, **kwargs):
    results = []
    initial = query(*args, **kwargs)
    perpage = len(initial['hits'])
    totalHits = min(500,initial['totalHits'])
    pages1 = int(np.floor(totalHits/perpage))
    for p in range(0, pages1):
        results.append(query(*args, **kwargs, page = p + 1))
    results.append(query(*args, **kwargs, page = pages1 + 1)) #TODO: fix redundant pages
    return results
    

def set_key(key):
    global api_key
    api_key = key

def update_cache(key, value):
    global cache
    cache[key] = (value, time.time())
    for k in cache.keys():
        tmp = cache[k]
        if isinstance(tmp, tuple):
            if time.time() - tmp[1] > cache_expiry:
                del cache[k]
    if time.time() - cache['last_saved'] > cache_update_interval:
        joblib.dump(cache, cache_path())
        cache['last_saved'] = time.time()