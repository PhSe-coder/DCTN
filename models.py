import os
from typing import List

from lightning.pytorch import LightningModule
from transformers import BertTokenizer
from transformers.modeling_outputs import TokenClassifierOutput

from constants import TAGS
from eval import absa_evaluate, evaluate
from model import BertForTokenClassification, FDGRModel
from optimization import BertAdam


class FDGRClassifer(LightningModule):

    def __init__(self,
                 num_labels: int,
                 output_dir: str,
                 lr: float,
                 alpha: float = 0.001,
                 beta: float = 0.01,
                 h_dim: int = 384,
                 pretrained_model_name: str = "bert-base-uncased"):
        """FDGR model classifier by pytorch lightning

        Parameters
        ----------
        num_labels : int
            number of tag labels
        output_dir : str
            the output directory of the annotation results
        lr : float
            learning rate
        alpha : float, optional
            weight for InfoNCE loss
        beta : float, optional
            weight for CLUB loss
        h_dim : int, optionl
            the dim of the disentangled features
        pretrained_model_name : str, optional
            the specific pretrained model
        """
        super(FDGRClassifer, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.lr = lr
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, model_max_length=100)
        self.model = FDGRModel.from_pretrained(pretrained_model_name, num_labels=self.num_labels, alpha=alpha,
                                               beta=beta, h_dim=h_dim, tokenizer=self.tokenizer)
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
        return BertAdam(pretrained_params, self.lr, 0.1, self.trainer.estimated_stepping_batches)

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        outputs = self.forward(**train_batch, log_dict=self.log_dict)
        loss = outputs.loss
        self.manual_backward(loss)
        opt.step()
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch["original"]
        targets = batch.pop("gold_labels")
        outputs: TokenClassifierOutput = self.forward(batch)
        logits = outputs.logits
        pred_list, gold_list = id2label(logits.detach().argmax(dim=-1).tolist(), targets.tolist())
        self.valid_out.append((pred_list, gold_list))
        return pred_list, gold_list

    def on_validation_epoch_end(self):
        gold_Y, pred_Y = [], []
        for pred_list, gold_list in self.valid_out:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
        val_pre, val_rec, val_f1 = absa_evaluate(pred_Y, gold_Y)
        self.log_dict({"val_pre": val_pre, "val_rec": val_rec, "val_f1": val_f1})
        self.valid_out.clear()

    def test_step(self, batch, batch_idx):
        batch = batch["original"]
        targets = batch.pop("gold_labels")
        outputs: TokenClassifierOutput = self.forward(batch)
        logits = outputs.logits
        pred_list, gold_list = id2label(logits.detach().argmax(dim=-1).tolist(), targets.tolist())
        sentence = self.tokenizer.batch_decode(batch.get("input_ids"), skip_special_tokens=True)
        self.test_out.append((pred_list, gold_list, sentence))
        return pred_list, gold_list, sentence

    def on_test_epoch_end(self) -> None:
        gold_Y, pred_Y, text = [], [], []
        for pred_list, gold_list, sentence in self.test_out:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
            text.extend(sentence)
        absa_test_pre, absa_test_rec, absa_test_f1 = absa_evaluate(pred_Y, gold_Y)
        self.log_dict({
            "absa_test_pre": round(absa_test_pre, 4),
            "absa_test_rec": round(absa_test_rec, 4),
            "absa_test_f1": round(absa_test_f1, 4)
        })
        ae_test_pre, ae_test_rec, ae_test_f1 = evaluate(pred_Y, gold_Y)
        self.log_dict({
            "ae_test_pre": round(ae_test_pre, 4),
            "ae_test_rec": round(ae_test_rec, 4),
            "ae_test_f1": round(ae_test_f1, 4)
        })
        if self.local_rank == 0:
            version = 0
            path = os.path.join(self.output_dir, str(version))
            while os.path.exists(path):
                version += 1
                path = os.path.join(self.output_dir, str(version))
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "predict.txt"), "w") as f:
                for i in range(len(gold_Y)):
                    f.write(f"{text[i]}***{' '.join(pred_Y[i])}***{' '.join(gold_Y[i])}\n")
            with open(os.path.join(path, "absa_prediction.txt"), "w") as f:
                content = f'test_pre: {absa_test_pre:.4f}, test_rec: {absa_test_rec:.4f}, test_f1: {absa_test_f1:.4f}'
                f.write(content)
            with open(os.path.join(path, "ae_prediction.txt"), "w") as f:
                content = f'test_pre: {ae_test_pre:.4f}, test_rec: {ae_test_rec:.4f}, test_f1: {ae_test_f1:.4f}'
                f.write(content)
        self.test_out.clear()


class BertClassifier(LightningModule):

    def __init__(self,
                 num_labels: int,
                 output_dir: str,
                 lr: float,
                 pretrained_model_name: str = "bert-base-uncased"):
        super(BertClassifier, self).__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.lr = lr
        self.automatic_optimization = False
        self.model = BertForTokenClassification.from_pretrained(pretrained_model_name,
                                                                num_labels=self.num_labels)
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
        pred_list, gold_list = id2label(logits.argmax(dim=-1).tolist(), targets.tolist())
        self.valid_out.append((pred_list, gold_list))
        return pred_list, gold_list

    def on_validation_epoch_end(self):
        gold_Y, pred_Y = [], []
        for pred_list, gold_list in self.valid_out:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
        val_pre, val_rec, val_f1 = absa_evaluate(pred_Y, gold_Y)
        self.log_dict({"val_pre": val_pre, "val_rec": val_rec, "val_f1": val_f1})
        self.valid_out.clear()

    def test_step(self, batch, batch_idx):
        batch = batch['original']
        targets = batch.pop("gold_labels")
        outputs: TokenClassifierOutput = self.forward(**batch)
        logits = outputs.logits
        pred_list, gold_list = id2label(logits.detach().argmax(dim=-1).tolist(), targets.tolist())
        sentence = self.tokenizer.batch_decode(batch.get("input_ids"), skip_special_tokens=True)
        self.test_out.append((pred_list, gold_list, sentence))
        return pred_list, gold_list, sentence

    def on_test_epoch_end(self) -> None:
        gold_Y, pred_Y, text = [], [], []
        for pred_list, gold_list, sentence in self.test_out:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
            text.extend(sentence)
        absa_test_pre, absa_test_rec, absa_test_f1 = absa_evaluate(pred_Y, gold_Y)
        self.log_dict({
            "absa_test_pre": round(absa_test_pre, 4),
            "absa_test_rec": round(absa_test_rec, 4),
            "absa_test_f1": round(absa_test_f1, 4)
        })
        ae_test_pre, ae_test_rec, ae_test_f1 = evaluate(pred_Y, gold_Y)
        self.log_dict({
            "ae_test_pre": round(ae_test_pre, 4),
            "ae_test_rec": round(ae_test_rec, 4),
            "ae_test_f1": round(ae_test_f1, 4)
        })
        if self.local_rank == 0:
            version = 0
            path = os.path.join(self.output_dir, str(version))
            while os.path.exists(path):
                version += 1
                path = os.path.join(self.output_dir, str(version))
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "predict.txt"), "w") as f:
                for i in range(len(gold_Y)):
                    f.write(f"{text[i]}***{' '.join(pred_Y[i])}***{' '.join(gold_Y[i])}\n")
            with open(os.path.join(path, "absa_prediction.txt"), "w") as f:
                content = f'test_pre: {absa_test_pre:.4f}, test_rec: {absa_test_rec:.4f}, test_f1: {absa_test_f1:.4f}'
                f.write(content)
            with open(os.path.join(path, "ae_prediction.txt"), "w") as f:
                content = f'test_pre: {ae_test_pre:.4f}, test_rec: {ae_test_rec:.4f}, test_f1: {ae_test_f1:.4f}'
                f.write(content)
        self.test_out.clear()


def id2label(predict: List[int], gold: List[List[int]]):
    gold_Y: List[List[str]] = []
    pred_Y: List[List[str]] = []
    for _gold in gold:
        gold_list = [TAGS[_gold[i]] for i in range(len(_gold)) if _gold[i] != -1]
        gold_Y.append(gold_list)
    idx = 0
    for item in gold_Y:
        pred_Y.append([TAGS[pred] for pred in predict[idx:idx + len(item)]])
        idx += len(item)
    assert idx == len(predict)
    return pred_Y, gold_Y
