import pyautogui
from sys import platform


def pressKey(k):
    """ Press the given key """
    if platform == "linux" or platform == "linux2":
        pyautogui.keyDown(k)
    elif platform == "win32":
        import pydirectinput
        pydirectinput.keyDown(k)


def releaseKey(k):
    """ Release the given key """
    if platform == "linux" or platform == "linux2":
        pyautogui.keyUp(k)
    elif platform == "win32":
        import pydirectinput
        pydirectinput.keyUp(k)


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
