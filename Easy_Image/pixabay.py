import python_pixabay
import tempfile
from sklearn.externals import joblib
import os
import time
import warnings
import json

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

def images_from_query(myquery, imgtype = "largeImageURL"):
    return [m[imgtype] for m in  myquery['hits']]

def query(*args, **kwargs):
    try:
        return cache[(args, json.dumps(kwargs))][0]
    except KeyError:
        update_api_key()
        pix = python_pixabay.Pixabay(api_key)
        results = pix.image_search(*args, **kwargs)
        update_cache((args, json.dumps(kwargs)), results)
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