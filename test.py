"""
Forked: github.com/qinnzou/Robust-Lane-Detection
"""

import time
import subprocess, signal, os

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

import config
from config import args_setting
from dataset import RoadSequenceDataset, RoadSequenceDatasetList
from model import generate_model


def output_result(model, test_loader, device, save_path: str = "./data/result"):
    """ Outputs the result and saves prediction as images """
    model.eval()
    k = 0
    feature_dic = []
    with torch.no_grad():
        for sample_batched in test_loader:
            k += 1
            data, _ = (
                sample_batched["data"].to(device),
                sample_batched["label"].type(torch.LongTensor).to(device),
            )
            output, feature = model(data)
            feature_dic.append(feature)
            pred = output.max(1, keepdim=True)[1]
            img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
            img = Image.fromarray(img.astype(np.uint8))

            data = torch.squeeze(data).cpu().numpy()
            if args.model == "SegNet-ConvLSTM" or "UNet-ConvLSTM":
                data = np.transpose(data[-1], [1, 2, 0]) * 255
            else:
                data = np.transpose(data, [1, 2, 0]) * 255
            data = Image.fromarray(data.astype(np.uint8))
            rows = img.size[0]
            cols = img.size[1]
            for i in range(0, rows):
                for j in range(0, cols):
                    img2 = img.getpixel((i, j))
                    if img2[0] > 200 or img2[1] > 200 or img2[2] > 200:
                        data.putpixel((i, j), (234, 53, 57, 255))
            data = data.convert("RGB")
            data.save(os.path.join(save_path, "%s_data.jpg" % k))  # redlines
            img.save(os.path.join(save_path, "%s_pred.jpg" % k))  # prediction

            data.show()
            time.sleep(0.5)
            p = subprocess.Popen(["ps", "-A"], stdout=subprocess.PIPE)
            out, _ = p.communicate()
            for line in out.splitlines():
                if b"eog" in line:
                    pid = int(line.split(None, 1)[0])
                    os.kill(pid, signal.SIGKILL)


def evaluate_model(model, test_loader, device, criterion):
    """ Evaluates the model """
    model.eval()
    i = 0
    precision = 0.0
    recall = 0.0
    test_loss = 0
    correct = 0
    error = 0
    with torch.no_grad():
        for sample_batched in test_loader:
            i += 1
            data, target = (
                sample_batched["data"].to(device),
                sample_batched["label"].type(torch.LongTensor).to(device),
            )
            output, _ = model(data)
            pred = output.max(1, keepdim=True)[1]  # the max value and the max index
            img = torch.squeeze(pred).cpu().numpy() * 255
            lab = torch.squeeze(target).cpu().numpy() * 255
            img = img.astype(np.uint8)
            lab = lab.astype(np.uint8)
            kernel = np.uint8(np.ones((3, 3)))

            # accuracy
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            # precision,recall,f1
            label_precision = cv2.dilate(lab, kernel)
            pred_recall = cv2.dilate(img, kernel)
            img = img.astype(np.int32)
            lab = lab.astype(np.int32)
            label_precision = label_precision.astype(np.int32)
            pred_recall = pred_recall.astype(np.int32)
            a = len(np.nonzero(img * label_precision)[1])
            b = len(np.nonzero(img)[1])
            if b == 0:
                error = error + 1
                continue
            else:
                precision += float(a / b)
            c = len(np.nonzero(pred_recall * lab)[1])
            d = len(np.nonzero(lab)[1])
            if d == 0:
                error = error + 1
                continue
            else:
                recall += float(c / d)
            f1_measure = (2 * precision * recall) / (precision + recall)

    test_loss /= len(test_loader.dataset) / args.test_batch_size
    test_acc = (
        100.0
        * int(correct)
        / (len(test_loader.dataset) * config.LABEL_HEIGHT * config.LABEL_WIDTH)
    )

    print(f"{config.CC_OKBLUE}Average loss:{config.CC_ENDC}", end="")
    print(f"{config.CC_BOLD}%.4f{config.CC_ENDC}" % test_loss)
    print(f"{config.CC_OKBLUE}Accuracy:{config.CC_ENDC}", end="")
    print(
        f"{config.CC_BOLD}%d/%d (%.4f){config.CC_ENDC}" % int(correct),
        len(test_loader.dataset),
        test_acc,
    )

    precision = precision / (len(test_loader.dataset) - error)
    recall = recall / (len(test_loader.dataset) - error)
    f1_measure = f1_measure / (len(test_loader.dataset) - error)

    print(f"{config.CC_OKBLUE}Precision:{config.CC_ENDC}", end="")
    print(f"{config.CC_BOLD}%.5f{config.CC_ENDC}" % precision)
    print(f"{config.CC_OKBLUE}Recall:{config.CC_ENDC}", end="")
    print(f"{config.CC_BOLD}%.5f{config.CC_ENDC}" % recall)
    print(f"{config.CC_OKBLUE}F1 Measure:{config.CC_ENDC}", end="")
    print(f"{config.CC_BOLD}%.5f{config.CC_ENDC}" % f1_measure)


def get_parameters(model, layer_name):
    modules_skipped = (nn.ReLU, nn.MaxPool2d, nn.Dropout2d, nn.UpsamplingBilinear2d)
    for name, module in model.named_children():
        if name in layer_name:
            for layer in module.children():
                if isinstance(layer, modules_skipped):
                    continue
                else:
                    for parma in layer.parameters():
                        yield parma


if __name__ == "__main__":
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    # load data for batches, num_workers for multiprocess
    if args.model == "SegNet-ConvLSTM" or "UNet-ConvLSTM":
        test_loader = torch.utils.data.DataLoader(
            RoadSequenceDatasetList(file_path=args.test_path, transforms=op_tranforms),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=1,
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            RoadSequenceDataset(file_path=args.test_path, transforms=op_tranforms),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=1,
        )

    # load model and weights
    model = generate_model(args)
    class_weight = torch.Tensor(config.CLASS_WEIGHT)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

    pretrained_dict = torch.load(config.PRETRAINED_PATH)
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)

    # output the result pictures
    if args.save_path == "":
        args.save_path = "./data/result"
    output_result(model, test_loader, device, args.save_path)

    # calculate the values of accuracy, precision, recall, f1_measure
    evaluate_model(model, test_loader, device, criterion)
