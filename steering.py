"""
Press keyboard buttons for steering in the game
"""

import config
from keys import left, right, straight


# THRESHOLD = # TODO add thresholds


def steer(direction: float = 0):
    if not config.KEY_EVENTS:
        return

    """ Steers to the given direction with thresholds """
    print(f"{config.CC_OKCYAN}Steering:{config.CC_ENDC}", end="")
    if direction > 2:
        print(f"{config.CC_BOLD}<<<{config.CC_ENDC}")
        left()
    elif direction < 0.5:
        print(f"{config.CC_BOLD}>>>{config.CC_ENDC}")
        right()
    else:
        print(f"{config.CC_BOLD}^^^{config.CC_ENDC}")
        straight()