import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertModel, BertPreTrainedModel
from constants import DEPREL_DICT, POS_DICT

from mi_estimators import CLUBMean, InfoNCE, jsd, vCLUB

logger = logging.getLogger(__name__)


@dataclass
class TokenClassifierOutput():
    loss: Optional[Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class FDGRPretrainedModel(BertPreTrainedModel):

    def __init__(self, config, h_dim: int):
        super(FDGRPretrainedModel, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.ha_dim = h_dim
        self.hc_dim = h_dim
        self.hidden_size: int = config.hidden_size
        # feature disentanglement module
        self.ha_encoder = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                        nn.ReLU(), nn.Linear(config.hidden_size, self.ha_dim),
                                        nn.ReLU(), nn.LayerNorm(self.ha_dim, 1e-12))
        self.hc_encoder = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                        nn.ReLU(), nn.Linear(config.hidden_size, self.hc_dim),
                                        nn.ReLU(), nn.LayerNorm(self.hc_dim, 1e-12))
        self.decoder = nn.Sequential(nn.Linear(self.ha_dim + self.hc_dim, config.hidden_size),
                                     nn.ReLU(), nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.ReLU(), nn.LayerNorm(config.hidden_size, 1e-12))
        self.mse = nn.MSELoss()
        self.mi_loss = InfoNCE(self.ha_dim, self.ha_dim)
        self.cross_mi_loss = InfoNCE(config.hidden_size, self.ha_dim)
        self.club_loss = CLUBMean(self.ha_dim, self.hc_dim)
        self.orthogonal_loss = vCLUB()

    def forward(self,
                original: Dict[str, Tensor],
                word_contrast: Dict[str, Tensor] = None,
                replace_index: Tensor = None,
                labeled: Tensor = None):
        assert torch.all(original['attention_mask'] == word_contrast['attention_mask']).item() == 1
        input_ids = torch.cat([original['input_ids'], word_contrast['input_ids']])
        token_type_ids = torch.cat([original['token_type_ids'], word_contrast['token_type_ids']])
        attention_mask = torch.cat([original['attention_mask'], word_contrast['attention_mask']])
        seq_output: Tensor = self.bert(input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask)[0]
        orig_seq_output, word_cont_seq_output = seq_output.chunk(2)
        ha = self.ha_encoder(seq_output)
        hc = self.hc_encoder(seq_output)
        # auto-encoder loss
        decoded = self.decoder(torch.cat([ha, hc], -1))
        reconstruct_loss = self.mse(
            seq_output.view(-1, self.hidden_size)[attention_mask.view(-1) == 1],
            decoded.view(-1, self.hidden_size)[attention_mask.view(-1) == 1])
        orig_ha, word_cont_ha = ha.chunk(2)
        orig_hc, word_cont_hc = hc.chunk(2)
        # orthogonal loss
        orthogonal_loss = self.orthogonal_loss.update(
            ha.view(-1, self.ha_dim)[attention_mask.view(-1) == 1],
            hc.view(-1, self.hc_dim)[attention_mask.view(-1) == 1])
        # attribute-specific representation loss
        active_mask = original['attention_mask'].view(-1) == 1
        mask = None
        count = active_mask.count_nonzero()
        o = original['input_ids'].view(-1)[active_mask]
        w = word_contrast['input_ids'].view(-1)[active_mask]
        mask = torch.empty(count, count, device=active_mask.device)
        for i in range(count):
            mask[i] = (o - w[i]) == 0
        mask = mask.mul(torch.eye(count, dtype=torch.int32, device=active_mask.device) ^ 1)
        ha_loss = self.mi_loss.learning_loss(
            orig_ha.view(-1, self.ha_dim)[active_mask],
            word_cont_ha.view(-1, self.ha_dim)[active_mask], mask)
        cross_mi_loss = self.cross_mi_loss.learning_loss(
            orig_seq_output.view(-1, self.hidden_size)[active_mask],
            word_cont_ha.view(-1, self.ha_dim)[active_mask], mask)
        cross_mi_loss += self.cross_mi_loss.learning_loss(
            word_cont_seq_output.view(-1, self.hidden_size)[active_mask],
            orig_ha.view(-1, self.ha_dim)[active_mask], mask)
        # cross_mi_loss += self.cross_mi_loss.learning_loss(
        #     orig_seq_output.view(-1, self.hidden_size)[active_mask],
        #     word_cont_hc.view(-1, self.ha_dim)[active_mask], mask)
        # cross_mi_loss += self.cross_mi_loss.learning_loss(
        #     word_cont_seq_output.view(-1, self.hidden_size)[active_mask],
        #     orig_hc.view(-1, self.hc_dim)[active_mask], mask)
        # cross_mi_loss *= 0.25
        # content-specific representation loss
        active_replace_mask = replace_index.view(-1) == 1
        inactive_replace_mask = replace_index.view(-1) == 0
        hc_loss_replaced = self.club_loss.learning_loss(
            orig_hc.view(-1, self.hc_dim)[active_mask & active_replace_mask],
            word_cont_hc.view(-1, self.hc_dim)[active_mask & active_replace_mask],
        )
        hc_loss_unreplaced = self.mse(
            orig_hc.view(-1, self.hc_dim)[active_mask & inactive_replace_mask],
            word_cont_hc.view(-1, self.hc_dim)[active_mask & inactive_replace_mask],
        )

        return TokenClassifierOutput(loss={
            "orthogonal_loss": orthogonal_loss,
            "reconstruct_loss": reconstruct_loss,
            "ha_loss": ha_loss,
            "cross_mi_loss": cross_mi_loss,
            "hc_loss_replaced": hc_loss_replaced,
            "hc_loss_unreplaced": hc_loss_unreplaced,
        },
                                     hidden_states=(orig_ha, orig_hc))


class FDGRModel(nn.Module):

    def __init__(self, num_labels: int, pretrained_path: str):
        super(FDGRModel, self).__init__()
        weights = torch.load(pretrained_path)
        pretrained_model_name = weights["hyper_parameters"]["pretrained_model_name"]
        h_dim = weights["hyper_parameters"]["h_dim"]
        self.fdgr = FDGRPretrainedModel.from_pretrained(pretrained_model_name, h_dim)
        self.fdgr.load_state_dict(
            {k.replace("model.", ''): v
             for k, v in weights['state_dict'].items()}, False)
        self.num_labels = num_labels
        self.mi_loss = InfoNCE(h_dim, num_labels)
        self.pos_embedding = nn.Embedding(len(POS_DICT), 30, 0)
        self.dep_embedding = nn.Embedding(len(DEPREL_DICT), 30, 0)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.classifier = nn.Linear(h_dim * 2 + 30 + 30, num_labels)

    def forward(self,
                original: Dict[str, Tensor],
                word_contrast: Dict[str, Tensor] = None,
                replace_index: Tensor = None,
                labeled: Tensor = None):
        outputs: TokenClassifierOutput = self.fdgr(original, word_contrast, replace_index, labeled)
        active_mask = original['valid_mask'].view(-1) == 1
        ha, hc = outputs.hidden_states
        pos = self.pos_embedding(original["pos_ids"])
        dep = self.dep_embedding(original["dep_ids"])
        hidden_state = torch.cat([ha, hc, pos, dep],
                                 -1).view(-1, self.classifier.in_features)[active_mask]
        logits: Tensor = self.classifier(torch.cat([ha, hc, pos, dep], -1)) / 2
        active_logits = logits.view(-1, self.num_labels)[active_mask]
        ce_loss = self.loss_fct(logits.view(-1, self.num_labels), original['gold_labels'].view(-1))
        outputs.loss["ce_loss"] = ce_loss
        pos_eye = torch.eye(self.pos_embedding.num_embeddings,
                            device=self.pos_embedding.weight.device)
        dep_eye = torch.eye(self.dep_embedding.num_embeddings,
                            device=self.dep_embedding.weight.device)
        outputs.loss["aux_loss"] = torch.norm(
            torch.mm(self.pos_embedding.weight, self.pos_embedding.weight.T) *
            (1 - pos_eye)) / (self.pos_embedding.num_embeddings ** 2)+ torch.norm(
                torch.mm(self.dep_embedding.weight, self.dep_embedding.weight.T) -
                (1 - dep_eye)) / (self.dep_embedding.num_embeddings ** 2)
        return TokenClassifierOutput(logits=active_logits,
                                     loss=outputs.loss,
                                     hidden_states=hidden_state)


class BertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self,
                input_ids: Tensor,
                token_type_ids: Tensor = None,
                attention_mask: Tensor = None,
                valid_mask: Tensor = None,
                gold_labels: Tensor = None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if gold_labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), gold_labels.view(-1))
        active_mask = valid_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_mask]
        hidden_states = sequence_output.view(-1, sequence_output.size(-1))[active_mask]
        return TokenClassifierOutput(logits=active_logits, loss=loss, hidden_states=hidden_states)