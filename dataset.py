import linecache
import torch
from typing import Dict, List, Tuple
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
            ann = anns[index]
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
            ann = anns[index]
            if offset == len(tokens[index]):
                index += 1
                offset = 0
            pos = DEPREL_DICT.get(ann, DEPREL_DICT.get('O'))
        dep_ids.append(pos)
    return dep_ids


def get_polarity(anns: List[str]) -> int:
    tag = {"T-NEG": -1, "T-NEU": 0, "T-POS": 1}
    for ann in anns:
        if ann != 'O':
            return tag[ann]
    raise ValueError("label list must contains a aspect term.")


class ModelDataset(Dataset):

    def __init__(self, filename: str, contrast_filename: str, vad_lexicon: str, target: str,
                 tokenizer: BertTokenizer):
        self.datafile = filename
        self.contrast_datafile = contrast_filename
        self.target = target
        self.total = sum(1 for _ in open(filename, "rb"))
        self.tokenizer = tokenizer
        self.training = filename.endswith(".train.txt")
        self.vad_laxicon: Dict[str, Tuple[float, float, float]] = {}
        with open(vad_lexicon, "r") as f:
            for line in f:
                word, v, a, d = line.split('\t')
                self.vad_laxicon[word] = (float(v), float(a), float(d))

    def process(self, text: str, anns: List[str], labels: List[str]) -> Dict[str, Tensor]:
        tok_dict: Dict[str, List[int]] = self.tokenizer(text,
                                                        padding=PaddingStrategy.MAX_LENGTH,
                                                        truncation=True)
        wordpiece_tokens = self.tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
        label_ids = transform(text, labels, wordpiece_tokens, self.tokenizer.all_special_tokens)
        polarity = get_polarity(labels)
        valid_mask = tok_dict.attention_mask.copy()
        valid_mask[0] = 0
        valid_mask[len(valid_mask) - valid_mask[::-1].index(1) - 1] = 0
        pos_anns = [ann.split('.')[0] for ann in anns]
        dep_anns = [ann.split('.')[1] for ann in anns]
        pos_ids = pos_transform(text.split(), pos_anns, wordpiece_tokens,
                                self.tokenizer.all_special_tokens, self.tokenizer.pad_token)
        dep_ids = dep_transform(text.split(), dep_anns, wordpiece_tokens,
                                self.tokenizer.all_special_tokens, self.tokenizer.pad_token)
        vad = [
            self.vad_laxicon.get(word, (0.5, 0.5, 0.5))
            for word in self.tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
        ]
        data = {
            "input_ids": as_tensor(tok_dict.input_ids),
            "aspect_ids": as_tensor(as_tensor(label_ids) > 0, dtype=torch.int32),
            "gold_labels": as_tensor(polarity),
            "attention_mask": as_tensor(tok_dict.attention_mask),
            "token_type_ids": as_tensor(tok_dict.token_type_ids),
            "valid_mask": as_tensor(valid_mask),
            "pos_ids": as_tensor(pos_ids),
            "dep_ids": as_tensor(dep_ids),
            "vad": as_tensor(vad)
        }
        return data

    def __getitem__(self, index):
        # `getline` method start from index 1 rather than 0
        line = linecache.getline(self.datafile, index + 1).strip()
        contrast_line = linecache.getline(self.contrast_datafile, index + 1).strip()
        text, annotations, gold_labels = line.rsplit("***", maxsplit=3)
        original = self.process(text, annotations, gold_labels.split())
        text, annotations, gold_labels = contrast_line.rsplit("***", maxsplit=3)
        contrast = self.process(text, annotations, gold_labels.split())
        return {"original": original, "contrast": contrast}

    def __len__(self):
        return self.total


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=100)
    dataset = ModelDataset("./processed/dataset/restaurant.train.txt", "./data/synonyms.json",
                           "./NRC-VAD-Lexicon.txt", "laptop", tokenizer)
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataloader = DataLoader(dataset, 16, False, num_workers=16)
    for batch in tqdm(dataloader):
        pass
