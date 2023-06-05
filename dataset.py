import json
import linecache
import random
import torch
import os.path as osp
from typing import Dict, List, Set, Tuple
from transformers import BertTokenizer
from torch import Tensor, as_tensor
from torch.utils.data import Dataset
from transformers.utils.generic import PaddingStrategy

from constants import POS_DICT, TAGS


def transform(
    text: str,
    text_labels: List[str],
    wordpiece_tokens: List[str],
    special_tokens: List[str],
) -> List[int]:
    text_tokens = text.split()
    assert len(text_tokens) == len(text_labels), text
    token_tuples = list(zip(text_tokens, text_labels))
    i, offset = 0, 0
    labels: List[str] = []
    for token in wordpiece_tokens:
        if token in special_tokens:
            tag = "SPECIAL_TOKEN"
        else:
            tt = token_tuples[i]
            if token.startswith("##"):
                tag = f"I{labels[-1][1:]}" if labels[-1] not in ["O", "SPECIAL_TOKEN"] else "O"
            else:
                tag = tt[1]
                if tag != "O":
                    tag = f"B{tag[1:]}" if labels[-1] in ["O", "SPECIAL_TOKEN"] else f"I{tag[1:]}"
            offset += len(token.replace("##", ""))
            if offset == len(tt[0]):
                i += 1
                offset = 0
        labels.append(tag)
    return [TAGS.index(label) if label in TAGS else -1 for label in labels]


def pos_transform(tokens: List[str], anns: List[str], wordpiece_tokens: List[str],
                  special_tokens: List[str], pad_token):
    index = 0
    pos_ids = []
    offset = 0
    for wordpiece_token in wordpiece_tokens:
        if wordpiece_token in special_tokens:
            if wordpiece_token == pad_token:
                pos = POS_DICT.get(pad_token)
            else:
                pos = POS_DICT.get('O')
        else:
            offset += len(wordpiece_token.replace("##", ''))
            ann = anns[index].split('.')[0]
            if offset == len(tokens[index]):
                index += 1
                offset = 0
            pos = POS_DICT.get(ann, POS_DICT.get('O'))
        pos_ids.append(pos)
    return pos_ids


class ModelDataset(Dataset):

    def __init__(self, filename: str, knowledge2token: str, token2knowledge: str, target: str,
                 tokenizer: BertTokenizer, labeled: bool=True):
        self.datafile = filename
        self.k2t: Dict[str, Dict[str, List[str]]] = json.load(open(knowledge2token, "r"))
        self.t2k: Dict[str, Dict[str, List[str]]] = json.load(open(token2knowledge, "r"))
        self.target = target
        self.domain = osp.basename(self.datafile).split('.')[0]
        self.total = sum(1 for _ in open(filename, "rb"))
        self.tokenizer = tokenizer
        self.training = filename.endswith(".train.txt")
        self.labeled = labeled

    def process(self, text: str, labels: List[str], anns: List[str]) -> Dict[str, Tensor]:
        tok_dict: Dict[str, List[int]] = self.tokenizer(text,
                                                        padding=PaddingStrategy.MAX_LENGTH,
                                                        truncation=True)
        wordpiece_tokens = self.tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
        labels = transform(text, labels, wordpiece_tokens, self.tokenizer.all_special_tokens)
        valid_mask = tok_dict.attention_mask.copy()
        valid_mask[0] = 0
        valid_mask[len(valid_mask) - valid_mask[::-1].index(1) - 1] = 0
        pos_ids = pos_transform(text.split(), anns, wordpiece_tokens, self.tokenizer.all_special_tokens,
                                self.tokenizer.pad_token)
        data = {
            "input_ids": as_tensor(tok_dict.input_ids),
            "gold_labels": as_tensor(labels),
            "attention_mask": as_tensor(tok_dict.attention_mask),
            "token_type_ids": as_tensor(tok_dict.token_type_ids),
            "valid_mask": as_tensor(valid_mask),
            "pos_ids": as_tensor(pos_ids)
        }
        return data

    def build_contrast(self, text: str,
                       annotations: List[str]) -> Tuple[List[int], List[int], List[int]]:
        tokens = text.split()
        contrast_tokens = tokens.copy()
        candidate_indices = [
            index for index, token in enumerate(tokens)
            if token in self.t2k[self.domain].keys() and index < 100
        ]
        indicies = []
        bpe_lens = []
        max_count = int(len(candidate_indices) * 0.2 + 1)
        while len(candidate_indices) != 0 and len(indicies) < max_count:
            index = random.choice(candidate_indices)
            domains = list(self.k2t.keys())
            domains.remove(self.domain)
            target = random.choice(domains)
            token_list = self.k2t[target].get(annotations[index])
            bpe_len = len(self.tokenizer.tokenize(contrast_tokens[index]))
            while token_list and len(indicies) < max_count:
                token = random.choice(token_list)
                if len(self.tokenizer.tokenize(token)) == bpe_len:
                    contrast_tokens[index] = token
                    indicies.append(index)
                    bpe_lens.append(bpe_len)
                token_list.remove(token)
            candidate_indices.remove(index)
        contrast_text = ' '.join(contrast_tokens)
        return contrast_text, indicies, bpe_lens

    def __getitem__(self, index):
        # `getline` method start from index 1 rather than 0
        line = linecache.getline(self.datafile, index + 1).strip()
        text, annotations, gold_labels = line.rsplit("***")
        tokens, ann_list = text.split(), annotations.split()
        contrast_text, indicies, bpe_lens = self.build_contrast(text, ann_list)
        original = self.process(text, gold_labels.split(), ann_list)
        word_contrast = self.process(contrast_text, gold_labels.split(), ann_list)
        replace_index = torch.zeros_like(original["input_ids"])
        one = torch.ones_like(original["input_ids"], dtype=torch.bool)
        zero = torch.zeros_like(original["input_ids"], dtype=torch.bool)
        if indicies:
            for i, bpe_len in zip(indicies, bpe_lens):
                start = len(self.tokenizer.tokenize(' '.join(tokens[:i]))) + 1
                replace_index[start:start + bpe_len] = 1
        return {
            "original": original,
            "word_contrast": word_contrast,
            "replace_index": replace_index,
            "labeled": one if self.labeled else zero
        }

    def func(self, tokens: List[str], rand_tokens: List[str], anns: List[str], rand_anns: List[str],
             ann_set: Set[str]):
        candidate_indices, candidate_rand_indices, bpe_lens = [], [], []
        for ann in ann_set:
            i, rand_i = anns.index(ann), rand_anns.index(ann)
            rand_tokens[rand_i] = tokens[i]
            candidate_indices.append(i)
            candidate_rand_indices.append(rand_i)
            bpe_lens.append(len(self.tokenizer.tokenize(tokens[i])))
        return candidate_indices, candidate_rand_indices, bpe_lens

    def __len__(self):
        return self.total


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=100)
    dataset = ModelDataset("./processed/dataset/restaurant.train.txt",
                           "./data/knowledge2token.json", "./data/token2knowledge.json", "laptop",
                           tokenizer)
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, 16, True)
    for batch in dataloader:
        print(batch)
