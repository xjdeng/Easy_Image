from PIL import Image
import json
import piexif
from piexif._exceptions import InvalidImageDataError

exif_field = 50707
"""
Uses the DNGBackwardVersion EXIF field to store strings and JSON. If you want
to use a different field, see: http://www.exiv2.org/tags.html
"""

def load(img):
    """
Note: img is the path to an image, not an EasyImage object!
    
Loads the information stored in the img EXIF under the EXIF field specified
in the variable exif_field into a JSON.
    """
    try:
        im = Image.open(img)
        raw = im._getexif()[exif_field].decode()
        return json.loads(raw)
    except (TypeError, OSError, KeyError):
        return None

def save(img, obj):
    """
Note: img is the path to an image, not an EasyImage object!    

Takes a serializable object in obj and stores it in the img under the EXIF
field specified in exif_field.
    """
    try:
        json_obj = json.dumps(obj)
        exif_bytes = piexif.dump({"0th":{exif_field:bytes(json_obj,"utf-8")}})
        piexif.insert(exif_bytes, img)
        return True
    except (TypeError, InvalidImageDataError):
        return False