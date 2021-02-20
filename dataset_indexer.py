"""
An index file generator for the generated dataset from ETS2.
This file can be used to train a model.
"""

import argparse
import os

import config
import utils


def index(folder_path: str):
    """
    Creates an index file for a dataset to be used with the model
    """

    index_file = os.path.join(folder_path, "index.txt")

    # find sub dirs to create the dataset index
    sub_dirs = [x[0] for x in os.walk(folder_path)]
    sub_dirs.sort(key=utils.natural_keys)
    sub_dirs = sub_dirs[1:]

    # remove the previous index
    try:
        os.remove(index_file)
    except:
        pass

    # loop in subdirs to add all images into the index file
    for sub_dir in sub_dirs:
        files = []
        for elem in os.scandir(sub_dir):
            if elem.path.endswith(".jpg") and elem.is_file():
                files.append(elem.path)
        files.sort(key=utils.natural_keys)
        with open(index_file, "a") as file:
            file.write(" ".join(files) + "\n")

    print(f"{config.CC_OKGREEN}Index created: %s{config.CC_ENDC}", index_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Indexer")
    parser.add_argument(
        "--folder",
        type=str,
        default="./data/testset/ets2",
        help="Type a folder which is created by dataset_creator.py",
        required=True,
    )
    args = parser.parse_args()

    index(args.folder)
