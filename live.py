from math import pow, sqrt, atan2, degrees
import signal
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from grab import grab
from torch.utils.data import Dataset


import config
from config import args_setting
from houghlinesp import hough_lines_p
from matrix import matrix
from model import generate_model
from utils import countdown, fna, fne, signal_handler


PATH = "./data/live/"
FILE = "live.jpg"


class SingleDataset(Dataset):
    def __init__(self, filename, transforms):
        self.img_list = []
        item = filename.strip().split()
        self.img_list.append(item)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = []
        data.append(
            torch.unsqueeze(self.transforms(Image.open(img_path_list[0])), dim=0)
        )
        data = torch.cat(data, 0)
        label = Image.open(img_path_list[0])
        label = torch.squeeze(self.transforms(label))
        sample = {"data": data, "label": label}
        return sample


def live(mode=0, continuous=False):
    """ Continuously redicts lanes in the game screen on gameplay """
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # argument overwrite
    mode = args.mode
    continuous = args.continuous

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    # load model and weights
    print(f"{config.CC_OKCYAN}Loading model... {config.CC_ENDC}", end="")
    model = generate_model(args)
    # class_weight = torch.Tensor(config.CLASS_WEIGHT)
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    pretrained_dict = torch.load(config.PRETRAINED_PATH)
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)
    print(f"{config.CC_OKGREEN}LOADED{config.CC_ENDC}")

    signal.signal(signal.SIGINT, signal_handler)  # exit signal check

    print(f"Mode: {config.CC_BOLD}%d{config.CC_ENDC}" % mode)
    print(f"Continuous: {config.CC_BOLD}%r{config.CC_ENDC}" % continuous)

    countdown()

    while True:
        img = grab()
        frame = fne(PATH + FILE)
        cv2.imwrite(frame, img)

        # load data for batches, num_workers for multiprocess
        test_loader = torch.utils.data.DataLoader(
            SingleDataset(frame, transforms=op_tranforms),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=1,
        )

        # continuous output
        model.eval()
        with torch.no_grad():
            for sample_batched in test_loader:
                data, _ = (
                    sample_batched["data"].to(device),
                    sample_batched["label"].type(torch.LongTensor).to(device),
                )
                output, _ = model(data)
                pred = output.max(1, keepdim=True)[1]

                # save first matrix in the tensor to a text file
                if len(pred) > 0 and len(pred[0]) > 0:
                    np.savetxt(
                        fna(frame, "matrix", "txt"), pred[0][0].cpu().numpy(), fmt="%d"
                    )

                # save predicted image
                img = (
                    torch.squeeze(pred).cpu().unsqueeze(2).expand(-1, -1, 3).numpy()
                    * 255
                )
                img = Image.fromarray(img.astype(np.uint8))
                predicted = fna(frame, "pred")
                img.save(predicted)

                if mode == 0:
                    hough_lines_p(predicted, frame)
                elif mode == 1:
                    matrix()
                else:
                    # TODO # add other methods to predict steering time, sensitivity and direction
                    pass

        if not continuous:
            break

        time.sleep(config.WAIT_FOR_NEXT_FRAME)


if __name__ == "__main__":
    live()
