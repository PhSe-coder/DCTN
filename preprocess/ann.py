import json
import logging
import pickle
import os.path as osp
from argparse import ArgumentParser
from glob import glob
from os import makedirs
import random
from typing import Dict, List, Tuple
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
from stanza_utils import annotation_plus

parser = ArgumentParser(description="Data annotation")
parser.add_argument("--src", type=str, default="./processed/tmp")
parser.add_argument("--dest", type=str, default="./processed/dataset")
parser.add_argument("--synonyms", type=str, default="./data/synonyms.json")
parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased")


def get_domain(file: str):
    return osp.basename(file).split('.')[0]


def build_contrast_sample(text: str, labels: List[str], tokenizer: BertTokenizer,
                          synonyms: Dict[str, Dict[str, List[List[str | int]]]],
                          domain: str) -> Tuple[List[int], List[int], List[int]]:
    tokens = text.split()
    contrast_tokens = tokens.copy()
    candidate_indices = [i for i, token in enumerate(tokens) if token in synonyms[domain].keys()]
    for index in candidate_indices:
        token_list = synonyms[domain].get(tokens[index])
        bpe_len = len(tokenizer.tokenize(contrast_tokens[index]))
        while token_list:
            token, weight = random.choice(token_list)
            if len(tokenizer.tokenize(token)) == bpe_len:
                contrast_tokens[index] = token
                labels[index] = ' '.join([labels[index]] * len(token.split()))
                break
            token_list.remove([token, weight])
    contrast_text = ' '.join(contrast_tokens)
    contrast_labels = ' '.join(labels)
    return contrast_text, contrast_labels


def generate_graph(sentence: List):
    return [w["head"] - 1 for w in sentence]


def annotate(file: str, dest: str, synonyms: Dict[str, Dict[str, List[List[str | int]]]],
             tokenizer: BertTokenizer):
    lines = [line for line in open(file, "r")]
    documents = [line.rsplit("***", maxsplit=1)[0] for line in lines]
    labels = [line.rsplit("***", maxsplit=1)[1].strip() for line in lines]
    sentences = annotation_plus(documents)
    domain = get_domain(file)
    results = [
        build_contrast_sample(text, gold_labels.split(), tokenizer, synonyms, domain)
        for text, gold_labels in zip(documents, labels)
    ]
    contrast_documents, contrast_labels = [r[0] for r in results], [r[1] for r in results]
    contrast_sentences = annotation_plus(contrast_documents)
    f = open(osp.join(dest, osp.split(file)[1]), "w")
    fname = osp.split(file)[1].split('.')
    fname_np = fname.copy()
    fname_np[-1] = "graph"
    fname.insert(1, "contrast")
    f1 = open(osp.join(dest, '.'.join(fname)), "w")
    l = len(sentences)
    id2graph: List[np.ndarray] = []
    contrast_id2graph: List[np.ndarray] = []
    f1name_np = fname.copy()
    f1name_np[-1] = "graph"
    fout = open(osp.join(dest, '.'.join(fname_np)), "wb")
    f1out = open(osp.join(dest, '.'.join(f1name_np)), "wb")
    for i in tqdm(range(l), desc=file, total=l):
        ann = ' '.join(f'{w["xpos"]}.{w["deprel"]}' for w in sentences[i].to_dict())
        cont_ann = ' '.join(f'{w["xpos"]}.{w["deprel"]}' for w in contrast_sentences[i].to_dict())
        f.write(f"{documents[i]}***{ann}***{labels[i]}\n")
        f1.write(f"{contrast_documents[i]}***{cont_ann}***{contrast_labels[i]}\n")
        id2graph.append(generate_graph(sentences[i].to_dict()))
        contrast_id2graph.append(generate_graph(contrast_sentences[i].to_dict()))
    f.close()
    f1.close()
    pickle.dump(id2graph, fout)
    pickle.dump(contrast_id2graph, f1out)
    fout.close()
    f1out.close()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    args = parser.parse_args()
    logger.info("Obtain the pos/deprel labels, and saving the results in %s", args.dest)
    makedirs(args.dest, exist_ok=True)
    synonyms: Dict[str, Dict[str, List[List[str | int]]]] = json.load(open(args.synonyms, 'r'))
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.pretrained_model,
                                                             model_max_length=128)
    for file in glob(osp.join(args.src, "*.txt")):
        annotate(file, args.dest, synonyms, tokenizer)
