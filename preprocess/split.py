import logging
import os.path as osp
from argparse import ArgumentParser
from glob import glob
from os import makedirs
from shutil import copy

import torch
from torch.utils.data import random_split

parser = ArgumentParser(description="Data split")
parser.add_argument("--src", type=str, default="./data/dataset")
parser.add_argument("--dest", type=str, default="./processed/tmp")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--ratio", type=float, default=0.7, help="split ratio of train set")


def split(file: str, rate=0.8, seed=42):
    with open(file, "r") as f:
        data = f.readlines()
    train_size = int(rate * len(data))
    return data[:train_size], data[train_size:]
    # test_size = len(data) - train_size
    # generator = torch.Generator().manual_seed(seed)
    # return random_split(data, [train_size, test_size], generator)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    args = parser.parse_args()
    logger.info("Split train set into train/validation dataset, and saving in %s", args.dest)
    makedirs(args.dest, exist_ok=True)
    for file in glob(osp.join(args.src, "*.train.txt")):
        train_set, valid_set = split(file, args.ratio, args.seed)
        with open(osp.join(args.dest, osp.split(file)[1]), "w") as f:
            f.writelines(train_set)
        with open(osp.join(args.dest, osp.split(file)[1].replace("train", "validation")), "w") as f:
            f.writelines(valid_set)
    logger.info("Copy test dataset into %s", args.dest)
    for file in glob(osp.join(args.src, "*.test.txt")):
        copy(file, args.dest)