from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import ModelDataset


@dataclass
class ABSADataModule(LightningDataModule):
    batch_size: int
    vad_lexicon_file: str
    affective_file: str
    num_workers: int = 0
    pretrained_model: str = "bert-base-uncased"
    train_file: str = None
    contrast_file: str = None
    validation_file: str = None
    test_file: str = None
    graph_suffix: str = ".graph"

    def __post_init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model, model_max_length=128)
        self.vad_laxicon: Dict[str, Tuple[float, float, float]] = {}
        with open(self.vad_lexicon_file, "r") as f:
            for line in f:
                word, v, a, d = line.split('\t')
                self.vad_laxicon[word] = (float(v), float(a), float(d))
        self.affective_space: Dict[str, list[float]] = {}
        with open(self.affective_file, "r", encoding="UTF-8-sig") as f:
            for line in f:
                items = line.strip().split(',')
                self.affective_space[items[0]] = [float(item) for item in items[1:]]
        self.dataloader = partial(DataLoader,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers)

    def setup(self, stage):
        if stage == 'fit':
            self.train_set = ModelDataset(self.train_file, self.vad_laxicon, self.tokenizer,
                                          self.affective_space, self.contrast_file,
                                          self.graph_suffix)
        if stage in ('fit', 'validate'):
            self.val_set = ModelDataset(self.validation_file,
                                        self.vad_laxicon,
                                        self.tokenizer,
                                        self.affective_space,
                                        graph_suffix=self.graph_suffix)
        if stage == 'test':
            self.test_set = ModelDataset(self.test_file,
                                         self.vad_laxicon,
                                         self.tokenizer,
                                         self.affective_space,
                                         graph_suffix=self.graph_suffix)

    def train_dataloader(self):
        return self.dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self.dataloader(self.test_set, shuffle=False)


@dataclass
class PretraninedABSADataModule(LightningDataModule):
    batch_size: int
    k2t_file: str
    t2k_file: str
    target: str
    num_workers: int = 0
    pretrained_model: str = "bert-base-uncased"
    source_train_file: str = None
    target_train_file: str = None
    validation_file: str = None
    test_file: str = None
    use_target: bool = False

    def __post_init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model, model_max_length=128)
        self.dataloader = partial(DataLoader,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers)

    def setup(self, stage):
        if stage == 'fit':
            source = ModelDataset(self.source_train_file, self.k2t_file, self.t2k_file, self.target,
                                  self.tokenizer)
            target = ModelDataset(self.target_train_file, self.k2t_file, self.t2k_file, self.target,
                                  self.tokenizer, False)
            # self.train_set = source if not self.use_target else target
            self.train_set = ModelDataset([self.source_train_file, self.target_train_file],
                                          self.k2t_file, self.t2k_file, self.target, self.tokenizer)

    def train_dataloader(self):
        return self.dataloader(self.train_set, shuffle=True)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ln = LightningCLI(save_config_kwargs={"overwrite": True}, run=False)
    ln.trainer.fit(ln.model, datamodule=ln.datamodule)
    ln.trainer.test(ln.model, ckpt_path="last", datamodule=ln.datamodule)
