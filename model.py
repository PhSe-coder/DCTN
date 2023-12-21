import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertModel, BertPreTrainedModel
from constants import DEPREL_DICT, POS_DICT

from mi_estimators import InfoNCE, vCLUB

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
        self.club_loss = vCLUB()
        self.v_layer = nn.Linear(self.hc_dim, self.hc_dim)
        self.a_layer = nn.Linear(self.hc_dim, self.hc_dim)
        self.d_layer = nn.Linear(self.hc_dim, self.hc_dim)
        self.proj_v = nn.Linear(self.hc_dim, 1)
        self.proj_a = nn.Linear(self.hc_dim, 1)
        self.proj_d = nn.Linear(self.hc_dim, 1)

    def forward(self, original: Dict[str, Tensor], contrast: Dict[str, Tensor] = None):
        if not self.training:
            input_ids = original['input_ids']
            token_type_ids = original['token_type_ids']
            attention_mask = original['attention_mask']
            seq_output: Tensor = self.bert(input_ids,
                                           token_type_ids=token_type_ids,
                                           attention_mask=attention_mask)[0]
            ha: Tensor = self.ha_encoder(seq_output)
            hc: Tensor = self.hc_encoder(seq_output)
            return TokenClassifierOutput(hidden_states=(ha, hc))
        assert torch.all(original['attention_mask'] == contrast['attention_mask']).item() == 1
        input_ids = torch.cat([original['input_ids'], contrast['input_ids']])
        token_type_ids = torch.cat([original['token_type_ids'], contrast['token_type_ids']])
        attention_mask = torch.cat([original['attention_mask'], contrast['attention_mask']])
        seq_output: Tensor = self.bert(input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask)[0]
        # orig_seq_output, cont_seq_output = seq_output.chunk(2)
        ha: Tensor = self.ha_encoder(seq_output)
        hc: Tensor = self.hc_encoder(seq_output)
        # 1. CLUB loss
        club_loss = self.club_loss.update(
            ha.view(-1, self.ha_dim)[attention_mask.view(-1) == 1],
            hc.view(-1, self.hc_dim)[attention_mask.view(-1) == 1])
        # 2. reconstruct loss
        decoded = self.decoder(torch.cat([ha, hc], -1))
        reconstruct_loss = self.mse(
            seq_output.view(-1, self.hidden_size)[attention_mask.view(-1) == 1],
            decoded.view(-1, self.hidden_size)[attention_mask.view(-1) == 1])
        orig_ha, cont_ha = ha.chunk(2)
        orig_hc, cont_hc = hc.chunk(2)
        # 3. attribute-specific representation loss
        active_mask = original['attention_mask'].view(-1) == 1
        count = active_mask.count_nonzero()
        o = original['input_ids'].view(-1)[active_mask]
        w = contrast['input_ids'].view(-1)[active_mask]
        mask = torch.empty(count, count, device=active_mask.device)
        for i in range(count):
            mask[i] = (o - w[i]) == 0
        mask = mask.mul(torch.eye(count, dtype=torch.int32, device=active_mask.device) ^ 1)
        ha_loss = self.mi_loss.learning_loss(
            orig_ha.view(-1, self.ha_dim)[active_mask],
            cont_ha.view(-1, self.ha_dim)[active_mask], mask)
        # 4. vad loss
        v, a, d = self.v_layer(hc), self.a_layer(hc), self.d_layer(hc)
        proj_v, proj_a, proj_d = self.proj_v(v), self.proj_a(a), self.proj_d(d)
        vad_loss = self.mse(torch.cat([proj_v, proj_a, proj_d], -1),
                            torch.cat([original["vad_ids"], contrast["vad_ids"]]))
        # 5. orthogonal loss
        stacked = torch.stack([v, a, d], 0)
        orthogonal_loss = torch.square(torch.norm(torch.mm(stacked, stacked.T) - torch.eye(3)))
        return TokenClassifierOutput(loss={
            "club_loss": club_loss,
            "reconstruct_loss": reconstruct_loss,
            "orthogonal_loss": orthogonal_loss,
            "ha_loss": ha_loss,
            "vad_loss": vad_loss
        },
                                     hidden_states=(orig_ha, orig_hc))


class FDGRModel(BertPreTrainedModel):

    def __init__(self, config, num_labels, h_dim):
        super(FDGRModel, self).__init__(config)
        self.fdgr = FDGRPretrainedModel(config, h_dim)
        self.num_labels = num_labels
        self.pos_embedding = nn.Embedding(len(POS_DICT), 30, 0)
        self.dep_embedding = nn.Embedding(len(DEPREL_DICT), 30, 0)
        self.classifier = nn.Linear(h_dim, num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, original: Dict[str, Tensor], contrast: Dict[str, Tensor] = None):
        outputs: TokenClassifierOutput = self.fdgr(original, contrast)
        valid_mask = original['valid_mask']
        ha, hc = outputs.hidden_states
        pos = self.pos_embedding(original["pos_ids"])
        dep = self.dep_embedding(original["dep_ids"])
        emb = torch.mul(hc, (valid_mask / valid_mask.sum(-1, True)).unsqueeze(-1)).sum(1)
        logits: Tensor = self.classifier(emb)
        if self.training:
            outputs.loss["ce_loss"] = self.loss_fct(
                logits, torch.cat(original['gold_labels'], contrast["gold_labels"]))
        return TokenClassifierOutput(logits=logits, loss=outputs.loss, hidden_states=emb)


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
                gold_labels: Tensor = None,
                aspect_ids: Tensor = None,
                pos_ids: Tensor = None,
                dep_ids: Tensor = None,
                vad_ids: Tensor = None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emb = torch.mul(sequence_output,
                        (valid_mask / valid_mask.sum(-1, True)).unsqueeze(-1)).sum(1)
        logits = self.classifier(emb)
        loss = None
        if gold_labels is not None:
            loss = self.loss_fct(logits, gold_labels)
        return TokenClassifierOutput(logits=logits, loss=loss, hidden_states=emb)
