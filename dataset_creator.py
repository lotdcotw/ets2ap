"""
Run this script while playing ETS2 to get screenshots in proper sizes as dataset
"""

import time
import os
from datetime import datetime

import cv2
import config
from grab import grab
import utils


def create(folder_path: str, split_at: int = 20):
    """ Starts creating the dataset in folder on gameplay """

    utils.countdown()

    folder_no = 1
    file_no = 1

    # create dataset group folder
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    while True:
        # screenshot
        img = grab()
        if img is None:
            break

        # path
        if file_no > split_at:
            folder_no += 1
            file_no = 1

        # folder path
        current_path = os.path.join(folder_path, str(folder_no))
        if not os.path.exists(current_path):
            os.mkdir(current_path)
        current_path = os.path.join(current_path, str(file_no) + ".jpg")
        file_no += 1

        print(f"{config.CC_OKCYAN}Written: %s{config.CC_ENDC}", current_path)
        cv2.imwrite(current_path, img)

        time.sleep(config.WAIT_FOR_NEXT_FRAME)


if __name__ == "__main__":
    now = datetime.now()
    folder = os.path.join(
        "./data", "testset", "ets2_" + now.strftime("%y-%m-%d_%H%M%S")
    )
    create(folder)
