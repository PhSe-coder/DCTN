import json
import linecache
import random
import os.path as osp
from typing import Dict, List, Tuple

from torch import Tensor, as_tensor
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.generic import PaddingStrategy

from constants import TAGS


def transform(
    text: str,
    text_labels: List[str],
    wordpiece_tokens: List[str],
    special_tokens: List[str],
) -> List[int]:
    text_tokens = text.split()
    assert len(text_tokens) == len(text_labels)
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


class ModelDataset(Dataset):

    def __init__(self, filename: str, knowledge2token: str, token2knowledge: str, target: str,
                 tokenizer: PreTrainedTokenizer):
        self.datafile = filename
        self.k2t: Dict[str, Dict[str, List[str]]] = json.load(open(knowledge2token, "r"))
        self.t2k: Dict[str, Dict[str, List[str]]] = json.load(open(token2knowledge, "r"))
        self.target = target
        self.domain = osp.basename(self.datafile).split('.')[0]
        self.total = sum(1 for _ in open(filename, "rb"))
        self.tokenizer = tokenizer

    def process(self, text: str, labels: List[str]) -> Dict[str, Tensor]:
        tok_dict: Dict[str, List[int]] = self.tokenizer(text,
                                                        padding=PaddingStrategy.MAX_LENGTH,
                                                        truncation=True)
        wordpiece_tokens = self.tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
        labels = transform(text, labels, wordpiece_tokens, self.tokenizer.all_special_tokens)
        data = {
            "input_ids": as_tensor(tok_dict.input_ids),
            "gold_labels": as_tensor(labels),
            "attention_mask": as_tensor(tok_dict.attention_mask),
            "token_type_ids": as_tensor(tok_dict.token_type_ids),
        }
        return data

    def build_contrast(self, text: str) -> Tuple[int, str]:
        tokens = text.split()
        candidate_indices = [index for index, token in enumerate(tokens) if token in self.t2k[self.domain].keys()]
        if not candidate_indices:
            return -1, text
        index = random.choice(candidate_indices)
        candidate_token = tokens[index]
        candidate_knowledges = self.t2k[self.domain][candidate_token]
        k_set = set()
        for k in candidate_knowledges:
            if k in self.k2t[self.target]:
                k_set.update(self.k2t[self.target][k])
        if k_set:
            token = random.choice(list(k_set))
        else:
            k = random.choice(candidate_knowledges)
            token = random.choice(self.k2t[self.target][f"NN.{k.split('.')[1]}"])
        contrast_text = f"{' '.join(tokens[:index])} {token} {' '.join(tokens[index+1:])}"
        return index, contrast_text

    def __getitem__(self, index):
        # `getline` method start from index 1 rather than 0
        line = linecache.getline(self.datafile, index + 1).strip()
        text, gold_labels = line.rsplit("***")
        original = self.process(text, gold_labels.split())
        index, contrast_text = self.build_contrast(text)
        contrast = self.process(contrast_text, gold_labels.split())
        return {"original": original, "contrast": contrast, "replace_index": index}

    def __len__(self):
        return self.total


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=100)
    dataset = ModelDataset("./processed/tmp/rest.train.txt",
                           "./data/knowledge2token.json",
                            "./data/token2knowledge.json",
                            "laptop",
                            tokenizer)
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, 16, True)
    for batch in dataloader:
        print(batch)