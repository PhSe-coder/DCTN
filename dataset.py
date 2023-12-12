import json
import linecache
import random
import torch
import os.path as osp
from typing import Dict, List, Set, Tuple, Union
from transformers import BertTokenizer
from torch import Tensor, as_tensor
from torch.utils.data import Dataset
from transformers.utils.generic import PaddingStrategy

from constants import POS_DICT, TAGS, DEPREL_DICT


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


def dep_transform(tokens: List[str], anns: List[str], wordpiece_tokens: List[str],
                  special_tokens: List[str], pad_token):
    index = 0
    dep_ids = []
    offset = 0
    for wordpiece_token in wordpiece_tokens:
        if wordpiece_token in special_tokens:
            if wordpiece_token == pad_token:
                pos = DEPREL_DICT.get(pad_token)
            else:
                pos = DEPREL_DICT.get('O')
        else:
            offset += len(wordpiece_token.replace("##", ''))
            ann = anns[index].split('.')[1]
            if offset == len(tokens[index]):
                index += 1
                offset = 0
            pos = DEPREL_DICT.get(ann, DEPREL_DICT.get('O'))
        dep_ids.append(pos)
    return dep_ids


class ModelDataset(Dataset):

    def __init__(self, filenames: Union[str, List[str]], synonyms: str, target: str,
                 tokenizer: BertTokenizer):
        if isinstance(filenames, str):
            filenames = [filenames]
        self.datafiles = filenames
        self.synonyms: Dict[str, Dict[str, List[List[str | int]]]] = json.load(open(synonyms, 'r'))
        self.target = target
        self.file_map = {filename: sum(1 for _ in open(filename, "rb")) for filename in filenames}
        self.total = min(self.file_map.values())
        self.tokenizer = tokenizer
        self.training = filenames[0].endswith(".train.txt")
        self.vad_laxicon: Dict[str, Tuple[float, float, float]] = {}
        with open("./NRC-VAD-Lexicon.txt", "r") as f:
            for line in f:
                word, v, a, d = line.split('\t')
                self.vad_laxicon[word] = (float(v), float(a), float(d))

    def process(self, text: str, labels: List[str], anns: List[str]) -> Dict[str, Tensor]:
        tok_dict: Dict[str, List[int]] = self.tokenizer(text,
                                                        padding=PaddingStrategy.MAX_LENGTH,
                                                        truncation=True)
        wordpiece_tokens = self.tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
        labels = transform(text, labels, wordpiece_tokens, self.tokenizer.all_special_tokens)
        valid_mask = tok_dict.attention_mask.copy()
        valid_mask[0] = 0
        valid_mask[len(valid_mask) - valid_mask[::-1].index(1) - 1] = 0
        pos_ids = pos_transform(text.split(), anns, wordpiece_tokens,
                                self.tokenizer.all_special_tokens, self.tokenizer.pad_token)
        dep_ids = dep_transform(text.split(), anns, wordpiece_tokens,
                                self.tokenizer.all_special_tokens, self.tokenizer.pad_token)
        vad = [self.vad_laxicon[word] for word in text.split()]
        data = {
            "input_ids": as_tensor(tok_dict.input_ids),
            "gold_labels": as_tensor(labels),
            "attention_mask": as_tensor(tok_dict.attention_mask),
            "token_type_ids": as_tensor(tok_dict.token_type_ids),
            "valid_mask": as_tensor(valid_mask),
            "pos_ids": as_tensor(pos_ids),
            "dep_ids": as_tensor(dep_ids),
            "vad": as_tensor(vad)
        }
        return data

    def build_contrast_sample(self, text: str,
                              domain: str) -> Tuple[List[int], List[int], List[int]]:
        tokens = text.split()
        contrast_tokens = tokens.copy()
        candidate_indices = [
            i for i, token in enumerate(tokens) if token in self.synonyms[domain].keys()
        ]
        indicies = []
        bpe_lens = []
        for index in candidate_indices:
            token_list = self.synonyms[domain].get(tokens[index])
            bpe_len = len(self.tokenizer.tokenize(contrast_tokens[index]))
            while token_list:
                token, weight = random.choice(token_list)
                if len(self.tokenizer.tokenize(token)) == bpe_len:
                    contrast_tokens[index] = token
                    indicies.append(index)
                    bpe_lens.append(bpe_len)
                token_list.remove(token)
        contrast_text = ' '.join(contrast_tokens)
        return contrast_text, indicies, bpe_lens

    def __getitem__(self, index):
        data = []
        for datafile in self.datafiles:
            # `getline` method start from index 1 rather than 0
            line = linecache.getline(datafile, index + 1).strip()
            text, annotations, gold_labels = line.rsplit("***", maxsplit=3)
            tokens, ann_list = text.split(), annotations.split()
            domain = osp.basename(datafile).split('.')[0]
            contrast_text, indicies, bpe_lens = self.build_contrast_sample(text, domain)
            original = self.process(text, gold_labels.split(), ann_list)
            word_contrast = self.process(contrast_text, gold_labels.split(), ann_list)
            replace_index = torch.zeros_like(original["input_ids"])
            if indicies:
                for i, bpe_len in zip(indicies, bpe_lens):
                    start = len(self.tokenizer.tokenize(' '.join(tokens[:i]))) + 1
                    replace_index[start:start + bpe_len] = 1
            data.append({
                "original": original,
                "word_contrast": word_contrast,
                "replace_index": replace_index
            })
        if len(data) == 1:
            return data[0]
        return data

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
