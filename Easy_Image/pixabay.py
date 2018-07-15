import python_pixabay
import tempfile
from path import Path as path
from sklearn.externals import joblib

cache_filename = "pixabay_cache.pkl"

def cache_path():
    return tempfile.gettempdir() + "/" + cache_filename

try:
    cache = joblib.load(cache_path())
    if isinstance(cache, dict) == False:
        cache = {}
except IOError:
    cache = {}