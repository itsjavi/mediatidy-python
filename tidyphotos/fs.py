import re, os
import pandas as pd
from PIL import Image, ExifTags
from datetime import datetime
import requests, zipfile, io

def get_images_recursive(path):
    path = os.path.abspath(path)
    pattern = re.compile(".*\.(jpg|jpeg|png)$", re.IGNORECASE) # TODO add support for HEIC images
    images = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if pattern.match(f):
                images.append(os.path.join(root, f))
    return images

def get_images_recursive(path):
    path = os.path.abspath(path)
    pattern = re.compile(".*\.(jpg|jpeg|png)$", re.IGNORECASE) # TODO add support for HEIC images
    images = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if pattern.match(f):
                images.append(os.path.join(root, f))
    return images
    
def get_file_creationtime(path):
    ctime = os.path.getctime(path) # Windows only
    if ctime is not None and ctime > 0:
        return ctime
    
    ctime = os.stat(path).st_birthtime # MacOS, some UNIX
    if ctime is not None and ctime > 0:
        return ctime
    
    return os.path.getmtime(path) # cross-platform, but it's the modification time

def timestamp_to_paths(ts):
    y = datetime.fromtimestamp(ts).strftime('%Y')
    ymd = datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S')
    
    return (y, ymd)

def get_images_metadata(images):
    meta = []
    
    for imgpath in images:
        img = Image.open(imgpath)
        exif = img.getexif()
        width, height = img.size
        
        if exif is None:
            continue
        
        exifdict = {
            ExifTags.TAGS[k]: v
            for k, v in exif.items()
            if k in ExifTags.TAGS
        }
        
        if "DateTime" in exifdict:
            ts = datetime.strptime(exifdict['DateTime'], "%Y:%m:%d %H:%M:%S").timestamp()
        else:
            ts = get_file_creationtime(imgpath)
            
        tspath = timestamp_to_paths(ts)
        
        meta.append({
            "path": imgpath,
            "width": width,
            "height": height,
            "cyear": tspath[0],
            "cdate": tspath[1],
            "ctime": ts
        })
        
        img.close()
    
    return pd.DataFrame(meta)

def extract_zip_from_url(url, dest):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dest)
