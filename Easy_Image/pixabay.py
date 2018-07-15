import python_pixabay
import tempfile
from path import Path as path
from sklearn.externals import joblib
import os
import warnings

cache_filename = "pixabay_cache.pkl"

def cache_path():
    return tempfile.gettempdir() + "/" + cache_filename

try:
    cache = joblib.load(cache_path())
    if isinstance(cache, dict) == False:
        cache = {}
except IOError:
    cache = {}

api_key = None

def update_api_key(): 
    try:
        api_key = os.environ['PIXABAY_API_KEY']
    except KeyError:
        message = """Pixabay API key not found.  Please get your key at 
https://pixabay.com/api/docs/ and either set it in your OS's PIXABAY_API_KEY 
environment variable or set it by calling the set_key() function.""".replace("\n","")
        warnings.warn(message)

def set_key(key):
    api_key = key

update_api_key()


