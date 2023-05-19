from dataclasses import dataclass
from typing import List
from lightning.pytorch import LightningDataModule
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import MyDataset
from functools import partial
from lightning.pytorch.cli import LightningCLI


@dataclass
class ABSADataModule(LightningDataModule):
    batch_size: int
    num_workers: int = 0
    pretrained_model = "bert-base-uncased"
    train_file: List[str] = None
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
            self.train_set = MyDataset(self.train_file, self.tokenizer)
        if stage in ('fit', 'validate'):
            self.val_set = MyDataset(self.validation_file, self.tokenizer)
        if stage == 'test':
            self.test_set = MyDataset(self.test_file, self.tokenizer)

    def train_dataloader(self):
        return self.dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self.dataloader(self.test_set, shuffle=False)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    LightningCLI(save_config_kwargs={"overwrite": True})