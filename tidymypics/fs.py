import hashlib
import re
import os
import sys
import pandas as pd
from PIL import Image, ExifTags
from datetime import datetime
import requests
import zipfile
import io
from . import utils as ut


def file_md5(fname):
    buffer_size = 4096
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(buffer_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_images_recursive(path):
    path = os.path.abspath(path)
    pattern = re.compile(".*\.(jpg|jpeg|png)$", re.IGNORECASE)
    images = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if pattern.match(f):
                images.append(os.path.join(root, f))
    return images


def get_file_creationtime(path):
    platform = sys.platform.lower()

    if "darwin" in platform or "linux" in platform:  # MacOS, some UNIX
        ctime = os.stat(path).st_birthtime
        if ctime is not None and ctime > 0:
            return ctime
    elif "win" in platform:  # Windows only
        ctime = os.path.getctime(path)
        if ctime is not None and ctime > 0:
            return ctime

    # cross-platform, but it's the modification time
    return os.path.getmtime(path)


def timestamp_to_paths(ts):
    y = datetime.fromtimestamp(ts).strftime('%Y')
    ymd = datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S')

    return (y, ymd)


def get_images_metadata(images):
    print(" - Reading EXIF data and calculating MD5 for every image...")
    meta = []

    imgc = len(images)
    imgn = 0

    ut.progress_bar(0, imgc)

    for imgpath in images:
        imgn += 1
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
            ts = datetime.strptime(
                exifdict['DateTime'], "%Y:%m:%d %H:%M:%S").timestamp()
        else:
            ts = get_file_creationtime(imgpath)

        tspath = timestamp_to_paths(ts)

        img.close()

        img_hash = file_md5(imgpath)

        meta.append({
            "path": imgpath,
            "width": width,
            "height": height,
            "cyear": tspath[0],
            "cdate": tspath[1],
            "ctime": ts,
            "md5hash": img_hash
        })
        ut.progress_bar(imgn, imgc)

    if len(meta) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(meta)

    return df.sort_values('path')


def extract_zip_from_url(url, dest):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dest)
