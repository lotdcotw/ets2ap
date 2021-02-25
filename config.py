"""
Forked: https://github.com/qinnzou/Robust-Lane-Detection
"""

import argparse

# Global parameters
DEFAULT_COUNTDOWN = 3

# Dataset setting
IMG_WIDTH = 256
IMG_HEIGHT = 128
IMG_CHANNEL = 3
LABEL_WIDTH = 256
LABEL_HEIGHT = 128
LABEL_CHANNEL = 1
CLASS_NUM = 2
WAIT_FOR_NEXT_FRAME = 0.05
DATA_LOADER_NUMWORKERS = 8

# Screen positioning
SCREEN_PAD_LEFT = 2  # add tolerance padding for the left side
SCREEN_PAD_TOP = 36  # title bar
SCREENSHOT_BOX = (
    SCREEN_PAD_LEFT,
    SCREEN_PAD_TOP,
    1280 + SCREEN_PAD_LEFT,
    720 + SCREEN_PAD_TOP,
)

# Paths
PRETRAINED_PATH = "./model/unetlstm.pth"

# Weight
CLASS_WEIGHT = [0.02, 1.02]

# Console colors
CC_HEADER = "\033[95m"
CC_OKBLUE = "\033[94m"
CC_OKCYAN = "\033[96m"
CC_OKGREEN = "\033[92m"
CC_WARNING = "\033[93m"
CC_ERROR = "\033[91m"
CC_ENDC = "\033[0m"
CC_BOLD = "\033[1m"
CC_UNDERLINE = "\033[4m"


def args_setting():
    """ Application arguments """

    parser = argparse.ArgumentParser(description="PyTorch UNet-ConvLSTM")
    parser.add_argument(
        "--model",
        type=str,
        default="UNet-ConvLSTM",
        help="( UNet-ConvLSTM | SegNet-ConvLSTM | UNet | SegNet | ",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        help="path of road data sequence files e.g. ./data/testset/ets2",
        default="./data/testset/ets2",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="path for prediction images e.g. ./data/result",
        default="./data/result",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        help="path for the training e.g. ./data/testset/ets2/train_index.txt",
        default="./data/testset/ets2/train_index.txt",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        help="path for the value index file e.g. ./data/testset/ets2/val_index.txt",
        default="./data/testset/ets2/val_index.txt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        metavar="N",
        help="input batch size for training (default: 10)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for testing (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        metavar="N",
        help="number of epochs to train (default: 30)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="use CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="(0: HoughLinesP, 1: Matrix)",
    )
    parser.add_argument(
        "--continuous",
        type=bool,
        default=False,
        help="Single shot or continuous capturing",
    )
    parser.add_argument(
        "--out",
        type=bool,
        default=False,
        help="Writes original and prediction images",
    )
    parser.add_argument(
        "--wff",
        type=float,
        default=1.0,
        help="Wait for frame in seconds (float)",
    )
    args = parser.parse_args()

    # overwrite parameters
    if args.wff is not None:
        global WAIT_FOR_NEXT_FRAME
        WAIT_FOR_NEXT_FRAME = args.wff

    return args
