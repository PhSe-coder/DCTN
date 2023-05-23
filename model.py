import logging
from typing import Dict
from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel, BertPreTrainedModel
from torch import Tensor
from transformers.modeling_outputs import TokenClassifierOutput
from mi_estimators import vCLUB, InfoNCE

logger = logging.getLogger(__name__)


class FDGRModel(BertPreTrainedModel):

    def __init__(self, config, alpha: float, beta: float):
        super(FDGRModel, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(0.1)
        self.hc_dim: int = config.hidden_size // 2
        self.ht_dim: int = config.hidden_size // 2
        self.register_buffer("alpha", torch.as_tensor(alpha))
        self.register_buffer("beta", torch.as_tensor(beta))
        # feature disentanglement module
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
                                 nn.Linear(config.hidden_size, self.hc_dim + self.ht_dim),
                                 nn.GELU(), nn.LayerNorm(self.hc_dim + self.ht_dim, 1e-12))
        self.decoder = nn.Sequential(nn.Linear(self.hc_dim + self.ht_dim, config.hidden_size),
                                     nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.GELU(), nn.LayerNorm(config.hidden_size, 1e-12))
        self.mse = nn.MSELoss()
        self.club_loss = vCLUB()
        self.mi_loss = InfoNCE(self.hc_dim, self.hc_dim)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.classifier = nn.Linear(self.hc_dim, config.num_labels)

    def forward(self,
                original: Dict[str, Tensor],
                contrast: Dict[str, Tensor] = None,
                replace_index: Tensor = None):
        loss = None
        if self.training:
            input_ids = torch.cat([original['input_ids'], contrast['input_ids']])
            token_type_ids = torch.cat([original['token_type_ids'], contrast['token_type_ids']])
            attention_mask = torch.cat([original['attention_mask'], contrast['attention_mask']])
            sequence_output = self.bert(input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)[0]
            mlp = self.mlp(sequence_output)
            hc, ht = torch.split(mlp, [self.hc_dim, self.ht_dim], dim=-1)
            logits: Tensor = self.classifier(hc)
            orig_logits, cont_logits = logits.chunk(2)
            active_mask = original['attention_mask'].view(-1) == 1
            active_orig_logits = orig_logits.view(-1, self.num_labels)[active_mask]
            # Standard cross entropy loss for source domain
            ce_loss = self.loss_fct(orig_logits.view(-1, self.num_labels),
                                    original['gold_labels'].view(-1))
            # Orthogonal loss
            orthogonal_loss = torch.matmul(
                hc.view(-1, self.hc_dim)[attention_mask.view(-1) == 1],
                ht.view(-1, self.ht_dim)[attention_mask.view(-1) == 1].T).abs().mean()
            # auto-encoder loss
            reconstruct_loss = self.mse(
                sequence_output.view(-1, sequence_output.size(-1))[attention_mask.view(-1) == 1],
                self.decoder(mlp).view(-1, sequence_output.size(-1))[attention_mask.view(-1) == 1])
            # token-invariant representation loss
            orig_ht, cont_ht = ht.chunk(2)
            active_replace_mask = replace_index.view(-1) == 1
            inactive_replace_mask = replace_index.view(-1) == 0
            replaced_token_loss = self.club_loss.update(
                orig_ht.view(-1, self.ht_dim)[active_mask & active_replace_mask],
                cont_ht.view(-1, self.ht_dim)[active_mask & active_replace_mask],
            )
            unreplaced_token_loss = self.mse(
                orig_ht.view(-1, self.ht_dim)[active_mask & inactive_replace_mask],
                cont_ht.view(-1, self.ht_dim)[active_mask & inactive_replace_mask],
            )
            token_loss = self.alpha * replaced_token_loss + unreplaced_token_loss
            # domain distribution loss
            orig_hc, cont_hc = hc.chunk(2)
            domain_loss = self.mi_loss.learning_loss(
                orig_hc.view(-1, self.hc_dim)[active_mask],
                cont_hc.view(-1, self.hc_dim)[active_mask],
            )
            # self.log_dict({
            #     "ce_loss": ce_loss.item(),
            #     "orthogonal_loss": orthogonal_loss.item(),
            #     "reconstruct_loss": reconstruct_loss.item(),
            #     "replaced_token_loss": replaced_token_loss.item(),
            #     "unreplaced_token_loss": unreplaced_token_loss.item(),
            #     "domain_loss": domain_loss.item(),
            # })
            loss = ce_loss + 0.01 * orthogonal_loss + 0.1 * reconstruct_loss + token_loss + self.beta * domain_loss
        else:
            input_ids = original['input_ids']
            attention_mask = original['attention_mask']
            token_type_ids = original['token_type_ids']
            valid_mask = original['valid_mask']
            sequence_output = self.bert(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)[0]
            mlp = self.mlp(sequence_output)
            hc, ht = torch.split(mlp, [self.hc_dim, self.ht_dim], dim=-1)
            orig_logits: Tensor = self.classifier(hc)
            active_mask = valid_mask.view(-1) == 1
            active_orig_logits = orig_logits.view(-1, self.num_labels)[active_mask]
        return TokenClassifierOutput(logits=active_orig_logits, loss=loss)


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