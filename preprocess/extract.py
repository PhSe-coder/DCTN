import json
import os.path as osp
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob

import fasttext
import fasttext.util
import numpy as np
from tqdm import tqdm

from stanza_utils import *

ALLOWED_POS = ("NN", "NNS", "NNP", "JJ")
ALLOWED_DEP = ("nsubj", "compound", "obj" ,"obl" ,"conj", "nmod", "amod", "root", "punct")

parser = ArgumentParser(description="Data split")
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--threshold", type=float, default=0.15)
parser.add_argument("--dest", type=str, default="./data")

def get_domain(file: str):
    return osp.basename(file).split('.')[0]

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    if np.all(vec2 == 0): return np.float32(0)
    return vec1.dot(vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        else:
            return super(Encoder, self).default(obj)

if __name__ == "__main__":
    data = {}
    tokens = {}
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')
    args = parser.parse_args()
    for file in glob(osp.join(args.data_dir, "**.train.txt")):
        documents = [line.split("***")[0] for line in open(file, "r")]
        sentences = annotation_plus(documents)
        domain = get_domain(file)
        domain_vec = ft.get_word_vector(domain)
        ret = defaultdict(set)
        t = defaultdict(set)
        for idx, sentence in tqdm(enumerate(sentences), desc=domain, total=len(sentences)):
            for token in sentence.to_dict():
                xpos, deprel = token["xpos"], token["deprel"]
                cos = cosine_similarity(domain_vec, ft.get_word_vector(token['text']))
                if xpos in ALLOWED_POS and deprel in ALLOWED_DEP and cos > args.threshold:
                    ret[f"{xpos}.{deprel}"].add(token['text'])
                    t[token['text']].add(f"{xpos}.{deprel}")
        data[domain] = ret
        tokens[domain] = t
    with open(osp.join(args.dest, "knowledge2token.json"), "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True, cls=Encoder)
    with open(osp.join(args.dest, "token2knowledge.json"), "w", encoding='utf-8') as f:
        json.dump(tokens, f, indent=2, sort_keys=True, cls=Encoder)
    