from dataclasses import dataclass
from functools import partial

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import ModelDataset


@dataclass
class ABSADataModule(LightningDataModule):
    batch_size: int
    k2t_file: str
    t2k_file: str
    target: str
    num_workers: int = 0
    pretrained_model: str = "bert-base-uncased"
    train_file: str = None
    validation_file: str = None
    test_file: str = None

    def __post_init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model, model_max_length=100)
        self.dataloader = partial(DataLoader,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers)

    def setup(self, stage):
        if stage == 'fit':
            self.train_set = ModelDataset(self.train_file, self.k2t_file, self.t2k_file,
                                          self.target, self.tokenizer)
        if stage in ('fit', 'validate'):
            self.val_set = ModelDataset(self.validation_file, self.k2t_file, self.t2k_file,
                                        self.target, self.tokenizer)
        if stage == 'test':
            self.test_set = ModelDataset(self.test_file, self.k2t_file, self.t2k_file, self.target,
                                         self.tokenizer)

    def train_dataloader(self):
        return self.dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self.dataloader(self.test_set, shuffle=False)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    LightningCLI(save_config_kwargs={"overwrite": True})