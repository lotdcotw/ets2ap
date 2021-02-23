from datetime import datetime
import os
import re
import sys
import time

import cv2
import numpy as np

import config
from keys import release_all


def atoi(text):
    """ try text to integer"""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


# countdown
def countdown(timer: int = config.DEFAULT_COUNTDOWN):
    """ countdown to focus on the game window and resume """
    print(
        f"{config.CC_WARNING}Focus on ETS2. Counting down...{config.CC_ENDC} ", end=""
    )
    for i in list(range(timer))[::-1]:
        print("{} ".format((i + 1)), end="")
        time.sleep(1)
        sys.stdout.flush()
    print()


def roi(img, vertices):
    """ Region of interest """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def signal_handler(sig, frame):
    """ Runs when exit signal is received """
    release_all()
    sys.exit(0)


def ts():
    """ Returns the current timestamp """
    now = datetime.now()
    return datetime.timestamp(now)


def fne(base_file_name, id=0):
    """ Returns a file name extension with timestamp and id/counter """
    stamp = ts()
    fn, ext = os.path.splitext(base_file_name)
    return fn + "_" + str(stamp) + "_" + str(id) + ext


def fna(file_name, addition, newExt=""):
    """ Adds string at the end of a filename """
    fn, ext = os.path.splitext(file_name)
    if newExt == "":
        return fn + "_" + addition + ext
    return fn + "_" + addition + "." + newExt
