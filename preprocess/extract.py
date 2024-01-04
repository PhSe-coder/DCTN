import json
import os.path as osp
from argparse import ArgumentParser
from collections import Counter
from glob import glob
from typing import List, Tuple
import requests
import fasttext
import fasttext.util
import nltk
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
import logging
from stanza_utils import *
from ratelimit import limits
from retry import retry

ALLOWED_POS = ("NN", "NNS", "NNP")
ALLOWED_DEP = ("nsubj", "compound", "obj", "obl", "conj", "nmod", "amod", "root", "punct")
nltk.download('stopwords')
parser = ArgumentParser(description="Data split")
parser.add_argument("--data-dir", type=str, default="./data/raw")
parser.add_argument("--threshold", type=float, default=0.3)
parser.add_argument("--dest", type=str, default="./data")


def get_domain(file: str):
    return osp.basename(file).split('.')[0]


def cos_sim(vec1: np.ndarray, vec2: np.ndarray):
    if np.all(vec2 == 0): return np.float32(0)
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def most_common_words(sentences: List[Sentence]) -> List[str]:
    stopword_sets = stopwords.words('english')
    ret = Counter()
    for sentence in sentences:
        words = [
            w for token in sentence.to_dict()
            if (w := token.get("lemma", token["text"]).lower()) not in stopword_sets
            and token["xpos"] in ALLOWED_POS and token["deprel"] in ALLOWED_DEP
        ]
        ret.update(Counter(words))
    return [word for word, count in ret.most_common(100)]


@limits(3600, 3600)
@retry(tries=10)
def get_synonym(word: str, low=0.5, high=0.8) -> List[Tuple[str, int]]:
    """Generate synonyms via ConceptNet5

    Parameters
    ----------
    word : str
        Input word
    low : float, optional
        Lower bound of the confidence interval, by default 0.5
    high : float, optional
        Upper bound of the confidence interval, by default 0.8

    Returns
    -------
    List[Tuple[str, int]]
        The (synonym, confidence) tuple list
    """
    response = requests.get(f'http://api.conceptnet.io/related/c/en/{word.lower()}?filter=/c/en')
    obj = response.json()
    return [(item["@id"].rsplit('/', maxsplit=1)[-1].replace('_', ' '), item["weight"])
            for item in obj["related"] if low < item["weight"] < high]


if __name__ == "__main__":
    data = {}
    # logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')
    args = parser.parse_args()
    for file in glob(osp.join(args.data_dir, "**.train.txt")):
        documents = [line.rsplit("***", maxsplit=1)[0] for line in open(file, "r")]
        docs = []
        for doc in documents:
            if doc not in docs:
                docs.append(doc)
        sentences = annotation_plus(docs)
        common_words = most_common_words(sentences)
        domain = get_domain(file)
        domain_vec = ft.get_word_vector(domain)
        domain_vecs = sorted([(word, ft.get_word_vector(word)) for word in common_words],
                             key=lambda items: cos_sim(domain_vec, items[1]),
                             reverse=True)[:10]
        synonym_dict = {}
        logger.info(f"Important words in {domain} domain:\n {[w for w, _ in domain_vecs]}")
        count = 0
        for idx, sentence in tqdm(enumerate(sentences), desc=domain, total=len(sentences)):
            for token in sentence.to_dict():
                xpos, deprel = token["xpos"], token["deprel"]
                is_sim = np.mean([
                    cos_sim(vec, ft.get_word_vector(token['text'])) for _, vec in domain_vecs
                ]) > args.threshold
                # Count the domain-specific words
                if xpos in ALLOWED_POS and deprel in ALLOWED_DEP and is_sim: count += 1
                # Generate synonyms fot the domain-specific words
                if xpos in ALLOWED_POS and deprel in ALLOWED_DEP and is_sim and (
                        token['text'] not in synonym_dict):
                    synonym_dict[token["text"]] = get_synonym(token["text"])
        data[domain] = synonym_dict
    with open(osp.join(args.dest, "synonyms.json"), "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
