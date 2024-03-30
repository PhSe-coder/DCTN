from dataclasses import dataclass
import linecache
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import BertTokenizer
from torch import Tensor, as_tensor
from torch.utils.data import Dataset
from transformers.utils.generic import PaddingStrategy
from constants import POS_DICT, TAGS, DEPREL_DICT


def transform(text: str, text_labels: List[str], wordpiece_tokens: List[str],
              special_tokens: List[str], unk_token: str) -> List[int]:
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
            if offset == len(tt[0]) or token == unk_token:
                i += 1
                offset = 0
        labels.append(tag)
    return [TAGS.index(label) if label in TAGS else -1 for label in labels]


def pos_transform(tokens: List[str], anns: List[str], wordpiece_tokens: List[str],
                  special_tokens: List[str], pad_token: str, unk_token: str):
    index = 0
    pos_ids = []
    offset = 0
    for w in wordpiece_tokens:
        if w in special_tokens:
            if w == pad_token:
                pos = POS_DICT.get(pad_token)
            else:
                pos = POS_DICT.get('O')
        else:
            offset += len(w.replace("##", ''))
            ann = anns[index]
            if offset == len(tokens[index]) or w == unk_token:
                index += 1
                offset = 0
            pos = POS_DICT.get(ann, POS_DICT.get('O'))
        pos_ids.append(pos)
    return pos_ids


def dep_transform(tokens: List[str], anns: List[str], wordpiece_tokens: List[str],
                  special_tokens: List[str], pad_token: str, unk_token: str):
    index = 0
    dep_ids = []
    offset = 0
    for w in wordpiece_tokens:
        if w in special_tokens:
            if w == pad_token:
                pos = DEPREL_DICT.get(pad_token)
            else:
                pos = DEPREL_DICT.get('O')
        else:
            offset += len(w.replace("##", ''))
            ann = anns[index]
            if offset == len(tokens[index]) or w == unk_token:
                index += 1
                offset = 0
            pos = DEPREL_DICT.get(ann, DEPREL_DICT.get('O'))
        dep_ids.append(pos)
    return dep_ids


def graph_transform(tokens: List[str], graph: List[int], wordpiece_tokens: List[str],
                    special_tokens: List[str], unk_token: str):
    l = len(wordpiece_tokens)
    graph_ids = np.zeros((l, l))
    index, offset = 0, 0
    ids: List[List[int]] = []
    for i, w in enumerate(wordpiece_tokens):
        if w in special_tokens: continue
        if offset == 0: ids.append([])
        offset += len(w.replace("##", ''))
        ids[index].append(i)
        if offset == len(tokens[index]) or w == unk_token:
            index += 1
            offset = 0
    index, offset = 0, 0
    for i, w in enumerate(wordpiece_tokens):
        if w in special_tokens: continue
        offset += len(w.replace("##", ''))
        gi = graph[index]
        if gi != -1:
            graph_ids[i, ids[gi]] = 1
            graph_ids[ids[gi], i] = 1
        if offset == len(tokens[index]):
            index += 1
            offset = 0
    return graph_ids


def vad_transform(tokens: List[str], vads: List[Tuple[float, float, float]],
                  default_vad: Tuple[float, float,
                                     float], wordpiece_tokens: List[str], special_tokens):
    index = 0
    vad_ids: List[Tuple[float, float, float]] = []
    offset = 0
    for wordpiece_token in wordpiece_tokens:
        if wordpiece_token in special_tokens:
            vad = default_vad
        else:
            offset += len(wordpiece_token.replace("##", ''))
            vad = vads[index]
            if offset == len(tokens[index]):
                index += 1
                offset = 0
        vad_ids.append(vad)
    return vad_ids


def affective_transform(tokens: List[str], affective: List[float], default_affective: List[float],
                        wordpiece_tokens: List[str], special_tokens):
    index = 0
    affective_embs: List[List[float]] = []
    offset = 0
    for wordpiece_token in wordpiece_tokens:
        if wordpiece_token in special_tokens:
            vad = default_affective
        else:
            offset += len(wordpiece_token.replace("##", ''))
            vad = affective[index]
            if offset == len(tokens[index]):
                index += 1
                offset = 0
        affective_embs.append(vad)
    return affective_embs


def get_polarity(anns: List[str]) -> int:
    tag = {"T-NEG": 0, "T-NEU": 1, "T-POS": 2}  # non-negative mapping
    for ann in anns:
        if ann != 'O':
            return tag[ann]
    raise ValueError("label list must contains a aspect term.")


@dataclass
class ModelDataset(Dataset):
    datafile: str
    vad_laxicon: Dict[str, Tuple[float, float, float]]
    tokenizer: BertTokenizer
    affective_space: Dict[str, List[float]]
    contrast_datafile: str = None
    graph_suffix: str = ".graph"

    def __post_init__(self):
        self.total = sum(1 for _ in open(self.datafile, "rb"))
        with open(str(Path(self.datafile).with_suffix(self.graph_suffix)), "rb") as f:
            self.id2head: List[int] = pickle.load(f)
        if self.contrast_datafile is not None:
            with open(str(Path(self.contrast_datafile).with_suffix(self.graph_suffix)), "rb") as f:
                self.contrast_id2head: List[int] = pickle.load(f)
        self.mean_affective_space = np.array(list(self.affective_space.values())).mean(axis=0).tolist()

    def process(self, text: str, anns: List[str], graph: List[int],
                labels: List[str]) -> Dict[str, Tensor]:
        tok_dict: Dict[str, List[int]] = self.tokenizer(text,
                                                        padding=PaddingStrategy.MAX_LENGTH,
                                                        truncation=True)
        wordpiece_tokens = self.tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
        spe_toks = self.tokenizer.all_special_tokens.copy()
        spe_toks.remove(self.tokenizer.unk_token)
        pad_token = self.tokenizer.pad_token
        unk_token = self.tokenizer.unk_token
        label_ids = transform(text, labels, wordpiece_tokens, spe_toks, unk_token)
        polarity = get_polarity(labels)
        valid_mask = tok_dict.attention_mask.copy()
        valid_mask[0] = 0
        valid_mask[len(valid_mask) - valid_mask[::-1].index(1) - 1] = 0
        pos_anns = [ann.rsplit('.', maxsplit=1)[0] for ann in anns]
        dep_anns = [ann.rsplit('.', maxsplit=1)[1] for ann in anns]
        pos_ids = pos_transform(text.split(), pos_anns, wordpiece_tokens, spe_toks, pad_token,
                                unk_token)
        dep_ids = dep_transform(text.split(), dep_anns, wordpiece_tokens, spe_toks, pad_token,
                                unk_token)
        graph_ids = graph_transform(text.split(), graph, wordpiece_tokens, spe_toks, unk_token)
        vads = [self.vad_laxicon.get(token, (0.5, 0.5, 0.5)) for token in text.split()]
        vad_ids = vad_transform(text.split(), vads, (0.5, 0.5, 0.5), wordpiece_tokens, spe_toks)
        affs = [
            self.affective_space.get(token, self.mean_affective_space) for token in text.split()
        ]
        affective_embeddings = affective_transform(text.split(), affs, self.mean_affective_space,
                                                   wordpiece_tokens, spe_toks)
        try:
            data = {
                "input_ids": as_tensor(tok_dict.input_ids),
                "token_type_ids": as_tensor(tok_dict.token_type_ids),
                "attention_mask": as_tensor(tok_dict.attention_mask),
                "valid_mask": as_tensor(valid_mask),
                "gold_labels": as_tensor(polarity),
                "aspect_ids": as_tensor(as_tensor(label_ids) > 0).int(),
                "span_indices": (as_tensor(label_ids) > 0).nonzero().squeeze(1)[[0,
                                                                                 -1]].unsqueeze(0),
                "pos_ids": as_tensor(pos_ids),
                "dep_ids": as_tensor(dep_ids),
                "graph_ids": torch.from_numpy(graph_ids),
                "vad_ids": as_tensor(vad_ids),
                "affective_ids": as_tensor(affective_embeddings)
            }
        except Exception as e:
            raise e
        return data

    def __getitem__(self, index):
        # `getline` method start from index 1 rather than 0
        line = linecache.getline(self.datafile, index + 1).strip()
        text, annotations, gold_labels = line.rsplit("***", maxsplit=2)
        original = self.process(text, annotations.split(), self.id2head[index], gold_labels.split())
        if self.contrast_datafile is not None:
            contrast_line = linecache.getline(self.contrast_datafile, index + 1).strip()
            text, annotations, gold_labels = contrast_line.rsplit("***", maxsplit=2)
            contrast = self.process(text, annotations.split(), self.contrast_id2head[index],
                                    gold_labels.split())
            return {"original": original, "contrast": contrast}
        else:
            return {"original": original}

    def __len__(self):
        return self.total


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=128)
    vad_laxicon: Dict[str, Tuple[float, float, float]] = {}
    with open("NRC-VAD-Lexicon.txt", "r") as f:
        for line in f:
            word, v, a, d = line.split('\t')
            vad_laxicon[word] = (float(v), float(a), float(d))
    dataset = ModelDataset("./processed/dataset/laptop.test.txt", vad_laxicon, tokenizer,
                           "./processed/dataset/laptop.test.txt")
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from lightning import seed_everything
    seed_everything(42)
    dataloader = DataLoader(dataset, 16, True, num_workers=0)
    for batch in tqdm(dataloader):
        pass
