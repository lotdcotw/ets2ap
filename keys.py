import pyautogui


def pressKey(k):
    """ Press the given key """
    pyautogui.keyDown(k)


def releaseKey(k):
    """ Release the given key """
    pyautogui.keyUp(k)


def straight():
    """ Throttle """
    pressKey("w")
    releaseKey("a")
    releaseKey("d")


def left():
    """ Turn left """
    pressKey("a")
    releaseKey("w")
    releaseKey("d")
    releaseKey("a")


def right():
    """ Turn right """
    pressKey("d")
    releaseKey("w")
    releaseKey("a")
    releaseKey("d")


def release_all():
    """ Release all keys """
    releaseKey("w")
    releaseKey("a")
    releaseKey("d")
