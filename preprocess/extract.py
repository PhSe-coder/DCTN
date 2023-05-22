import json
import os.path as osp
from argparse import ArgumentParser
from collections import Counter, defaultdict
from glob import glob
from typing import List

import fasttext
import fasttext.util
import nltk
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm

from stanza_utils import *

ALLOWED_POS = ("NN", "NNS", "NNP")
ALLOWED_DEP = ("nsubj", "compound", "obj", "obl", "conj", "nmod", "amod", "root", "punct")
# nltk.download('stopwords')
parser = ArgumentParser(description="Data split")
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--threshold", type=float, default=0.3)
parser.add_argument("--dest", type=str, default="./data")


def get_domain(file: str):
    return osp.basename(file).split('.')[0]


def cos_sim(vec1: np.ndarray, vec2: np.ndarray):
    if np.all(vec2 == 0): return np.float32(0)
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class Encoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        else:
            return super(Encoder, self).default(obj)


def most_common_words(sentences: List[Sentence]) -> List[str]:
    stopword_sets = stopwords.words('english')
    ret = Counter()
    for sentence in sentences:
        words = [
            token.get("lemma", token["text"]).lower() for token in sentence.to_dict()
            if token.get("lemma", token["text"]).lower() not in stopword_sets
            and token["xpos"] in ALLOWED_POS and token["deprel"] in ALLOWED_DEP
        ]
        ret.update(Counter(words))
    return [word for word, count in ret.most_common(100)]


if __name__ == "__main__":
    data = {}
    tokens = {}
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')
    args = parser.parse_args()
    for file in glob(osp.join(args.data_dir, "**.train.txt")):
        documents = [line.split("***")[0] for line in open(file, "r")]
        sentences = annotation_plus(documents)
        common_words = most_common_words(sentences)
        domain = get_domain(file)
        domain_vec = ft.get_word_vector(domain)
        domain_vecs = sorted([(word, ft.get_word_vector(word)) for word in common_words],
                             key=lambda items: cos_sim(domain_vec, items[1]),
                             reverse=True)[:10]
        ret = defaultdict(set)
        t = defaultdict(set)
        for idx, sentence in tqdm(enumerate(sentences), desc=domain, total=len(sentences)):
            for token in sentence.to_dict():
                xpos, deprel = token["xpos"], token["deprel"]
                is_sim = any(
                    cos_sim(vec, ft.get_word_vector(token['text'])) > args.threshold
                    for word, vec in domain_vecs)
                if xpos in ALLOWED_POS and deprel in ALLOWED_DEP and is_sim:
                    ret[f"{xpos}.{deprel}"].add(token['text'])
                    t[token['text']].add(f"{xpos}.{deprel}")
        data[domain] = ret
        tokens[domain] = t
    with open(osp.join(args.dest, "knowledge2token.json"), "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True, cls=Encoder)
    with open(osp.join(args.dest, "token2knowledge.json"), "w", encoding='utf-8') as f:
        json.dump(tokens, f, indent=2, sort_keys=True, cls=Encoder)
