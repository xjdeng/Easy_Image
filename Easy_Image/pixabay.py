import python_pixabay
import tempfile
from path import Path as path
from sklearn.externals import joblib
import os
import time
import warnings

cache_filename = "pixabay_cache.pkl"

api_key = None
cache_update_interval = 3600
cache_expiry = 24*3600

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
    try:
        api_key = os.environ['PIXABAY_API_KEY']
    except KeyError:
        message = """Pixabay API key not found.  Please get your key at 
https://pixabay.com/api/docs/ and either set it in your OS's PIXABAY_API_KEY 
environment variable or set it by calling the set_key() function.""".replace("\n","")
        warnings.warn(message)

update_api_key()

def set_key(key):
    api_key = key

def update_cache(key, value):
    cache[key] = (value, time.time())
    for k in cache.keys():
        tmp = cache[k]
        if isinstance(tmp, tuple):
            if time.time() - tmp[1] > cache_expiry:
                del cache[k]
    if time.time() - cache['last_saved'] > cache_update_interval:
        joblib.dump(cache, cache_path())
        cache['last_saved'] = time.time()