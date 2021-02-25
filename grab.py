import numpy as np
import pyscreenshot as ImageGrab
import cv2

import config


def grab():
    """
    grab a scaled and cut image / compatible image for the model: 256 x 128
    """

    # take a screenshot
    # zero childprocess and mss gives the best performance in most cases
    # adjust your window for ETS2 screen coordinates in config.py
    img = ImageGrab.grab(bbox=config.SCREENSHOT_BOX, backend="mss", childprocess=False)

    # color conversion
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # resize to expected model size
    # depending on your model, or training data, adjust it in config.py
    ratio = config.IMG_WIDTH / (config.SCREENSHOT_BOX[2] - config.SCREENSHOT_BOX[0])
    dsize = (
        config.IMG_WIDTH,
        int((config.SCREENSHOT_BOX[3] - config.SCREENSHOT_BOX[1]) * ratio),
    )
    img = cv2.resize(img, dsize=dsize)
    # crop to model size
    height_diff = dsize[1] - config.IMG_HEIGHT
    if height_diff < 0:
        print(
            f"{config.CC_ERROR}Error: Invalid screen size configuraion{config.CC_ENDC}"
        )
        return None
    img = img[height_diff : height_diff + config.IMG_HEIGHT, 0 : config.IMG_WIDTH]

    return img