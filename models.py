import torch
from lightning.pytorch import LightningModule
from transformers import BertTokenizer, BertConfig
from transformers.modeling_outputs import TokenClassifierOutput
from torch import as_tensor
from model import BertForTokenClassification, FDGRModel
from optimization import BertAdam
from torchmetrics.classification import MulticlassF1Score

class LossWeight:

    @staticmethod
    def weight1(batch_rate: float, tau=0.1):
        return tau * torch.sigmoid(torch.as_tensor(20 * (batch_rate - 3.0 / 4))) + 0.001

    @staticmethod
    def weight2(batch_rate: float, tau=0.1):
        return tau * torch.sigmoid(torch.as_tensor(20 * (batch_rate - 3.0 / 4))) + 0.001


class FDGRClassifer(LightningModule, LossWeight):

    def __init__(self,
                 num_labels: int,
                 lr: float,
                 coff: float = 0.02,
                 h_dim: int = 300,
                 pretrained_model_name: str = "bert-base-uncased"):
        """FDGR model classifier by pytorch lightning

        Parameters
        ----------
        num_labels : int
            number of tag labels
        lr : float
            learning rate
        coff: float
            the weight of the sub loss
        h_dim : int, optionl
            the dim of the disentangled features
        pretrained_model_name : str, optional
            the specific pretrained model
        """
        super(FDGRClassifer, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.num_labels = num_labels
        self.lr = lr
        self.coff = coff
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, model_max_length=100)
        config = BertConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)
        config.name_or_path = pretrained_model_name
        self.model = FDGRModel(config, h_dim)
        self.valid_out = []
        self.test_out = []

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        pretrained_param_optimizer = [(k, v) for k, v in self.named_parameters()
                                      if v.requires_grad == True and 'pooler' not in k]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        pretrained_params = [{
            'params':
            [p for n, p in pretrained_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
            1e-2
        }, {
            'params': [p for n, p in pretrained_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        return BertAdam(pretrained_params, self.lr, 0.0, self.trainer.estimated_stepping_batches)

    def training_step(self, train_batch, batch_idx):
        batch_rate: float = (1.0 * batch_idx + self.trainer.num_training_batches *
                             self.trainer.current_epoch) / self.trainer.estimated_stepping_batches
        opt = self.optimizers()
        opt.zero_grad()
        outputs = self.forward(**train_batch)
        ce_loss = outputs.loss["ce_loss"]
        orthogonal_loss = outputs.loss["orthogonal_loss"]
        reconstruct_loss = outputs.loss["reconstruct_loss"]
        ha_loss = outputs.loss["ha_loss"]
        club_loss = outputs.loss["club_loss"]
        vad_loss = outputs.loss["vad_loss"]
        # sub_loss = orthogonal_loss + reconstruct_loss + ha_loss + club_loss + vad_loss
        sub_loss = 0.01 * reconstruct_loss + 0.01 * ha_loss + 0.001 * club_loss + 0.01 * vad_loss
        loss = ce_loss +  sub_loss
        self.manual_backward(loss)
        opt.step()
        self.log('train_loss', loss.item())
        self.log_dict({
            "ce_loss": ce_loss.item(),
            "club_loss": club_loss.item(),
            "orthogonal_loss": orthogonal_loss.item(),
            "reconstruct_loss": reconstruct_loss.item(),
            "ha_loss": ha_loss.item(),
            "vad_loss": vad_loss.item()
        })
        return loss

    def validation_step(self, batch, batch_idx):
        outputs: TokenClassifierOutput = self.forward(**batch)
        batch = batch["original"]
        targets = batch.pop("gold_labels")
        logits = outputs.logits
        pred_list, gold_list = logits.argmax(dim=-1).tolist(), targets.tolist()
        self.valid_out.append((pred_list, gold_list))
        return pred_list, gold_list

    def on_validation_epoch_end(self):
        gold_Y, pred_Y = [], []
        for pred_list, gold_list in self.valid_out:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
        val_f1 = MulticlassF1Score(self.num_labels)(as_tensor(pred_Y), as_tensor(gold_Y))
        self.log("val_f1", val_f1)
        self.valid_out.clear()

    def test_step(self, batch, batch_idx):
        outputs: TokenClassifierOutput = self.forward(**batch)
        batch = batch["original"]
        targets = batch.pop("gold_labels")
        logits = outputs.logits
        pred_list, gold_list = logits.argmax(dim=-1).tolist(), targets.tolist()
        self.test_out.append((pred_list, gold_list))
        return pred_list, gold_list

    def on_test_epoch_end(self):
        gold_Y, pred_Y = [], []
        for pred_list, gold_list in self.test_out:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
        test_f1 = MulticlassF1Score(self.num_labels)(as_tensor(pred_Y), as_tensor(gold_Y))
        self.log("test_f1", test_f1)
        self.test_out.clear()


class BertClassifier(LightningModule):

    def __init__(self,
                 num_labels: int,
                 lr: float,
                 pretrained_model_name: str = "bert-base-uncased"):
        super(BertClassifier, self).__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.lr = lr
        self.automatic_optimization = False
        self.model = BertForTokenClassification(pretrained_model_name, num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, model_max_length=100)
        self.valid_out = []
        self.test_out = []

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        pretrained_param_optimizer = [(k, v) for k, v in self.named_parameters()
                                      if 'pooler' not in k]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params = [{
            'params':
            [p for n, p in pretrained_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
            1e-2
        }, {
            'params': [p for n, p in pretrained_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        return BertAdam(params, self.lr, 0.1, self.trainer.estimated_stepping_batches)

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        outputs = self.forward(**train_batch['original'])
        loss = outputs.loss
        self.manual_backward(loss)
        opt.step()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch['original']
        targets = batch.pop("gold_labels")
        outputs: TokenClassifierOutput = self.forward(**batch)
        logits = outputs.logits
        pred_list, gold_list = logits.argmax(dim=-1).tolist(), targets.tolist()
        self.valid_out.append((pred_list, gold_list))
        return pred_list, gold_list

    def on_validation_epoch_end(self):
        gold_Y, pred_Y = [], []
        for pred_list, gold_list in self.valid_out:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
        val_f1 = MulticlassF1Score(self.num_labels)(as_tensor(pred_Y), as_tensor(gold_Y))
        self.log_dict({"val_f1": val_f1})
        self.valid_out.clear()

    def test_step(self, batch, batch_idx):
        batch = batch['original']
        targets = batch.pop("gold_labels")
        outputs: TokenClassifierOutput = self.forward(**batch)
        logits = outputs.logits
        pred_list, gold_list = logits.argmax(dim=-1).tolist(), targets.tolist()
        self.test_out.append((pred_list, gold_list))
        return pred_list, gold_list

    def on_test_epoch_end(self) -> None:
        gold_Y, pred_Y = [], []
        for pred_list, gold_list in self.test_out:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
        f1 = MulticlassF1Score(self.num_labels)(as_tensor(pred_Y), as_tensor(gold_Y))
        self.log_dict({"test_f1": f1})
        self.test_out.clear()
