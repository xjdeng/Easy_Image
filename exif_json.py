from PIL import Image
import json
import piexif
from piexif._exceptions import InvalidImageDataError

exif_field = 40092

def load(img):
    try:
        im = Image.open(img)
        raw = im._getexif()[exif_field].decode()
        return json.loads(raw)
    except (TypeError, OSError, KeyError):
        return None

def save(img, obj):
    try:
        json_obj = json.dumps(obj)
        exif_bytes = piexif.dump({"0th":{exif_field:bytes(json_obj,"utf-8")}})
        piexif.insert(exif_bytes, img)
        return True
    except (TypeError, InvalidImageDataError):
        return False