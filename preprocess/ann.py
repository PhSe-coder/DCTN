import logging
import os.path as osp
from argparse import ArgumentParser
from glob import glob
from os import makedirs
from shutil import copy

from tqdm import tqdm

from stanza_utils import annotation_plus

parser = ArgumentParser(description="Data annotation")
parser.add_argument("--src", type=str, default="./processed/tmp")
parser.add_argument("--dest", type=str, default="./processed/dataset")


def get_domain(file: str):
    return osp.basename(file).split('.')[0]


def annotate(file: str, dest: str):
    lines = [line for line in open(file, "r")]
    documents = [line.split("***")[0] for line in lines]
    labels = [line.split("***")[1] for line in lines]
    sentences = annotation_plus(documents)
    with open(osp.join(dest, osp.split(file)[1]), "w") as f:
        for idx, sentence in tqdm(enumerate(sentences), desc=file, total=len(sentences)):
            anno = ' '.join(f'{token["xpos"]}.{token["deprel"]}' for token in sentence.to_dict())
            f.write(f"{documents[idx]}***{anno}***{labels[idx]}")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    args = parser.parse_args()
    logger.info("annotate train set to obtain the pos/deprel labels, save the results in %s",
                args.dest)
    makedirs(args.dest, exist_ok=True)
    for file in glob(osp.join(args.src, "*.txt")):
        annotate(file, args.dest)