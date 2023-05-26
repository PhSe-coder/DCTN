import logging
from typing import Dict
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from torch import Tensor
from transformers.modeling_outputs import TokenClassifierOutput
from mi_estimators import jsd, InfoNCE

logger = logging.getLogger(__name__)


class FDGRModel(BertPreTrainedModel):

    def __init__(self, config, alpha: float, beta: float, h_dim: int, tokenizer: BertTokenizer):
        super(FDGRModel, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(0.1)
        self.hc_dim = h_dim
        self.ht_dim = h_dim
        self.register_buffer("alpha", torch.as_tensor(alpha))
        self.register_buffer("beta", torch.as_tensor(beta))
        self.ht_project = nn.Linear(config.hidden_size, h_dim)
        # feature disentanglement module
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
                                 nn.Linear(config.hidden_size, self.hc_dim + self.ht_dim),
                                 nn.GELU(), nn.LayerNorm(self.hc_dim + self.ht_dim, 1e-12))
        self.decoder = nn.Sequential(nn.Linear(self.hc_dim + self.ht_dim, config.hidden_size),
                                     nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.GELU(), nn.LayerNorm(config.hidden_size, 1e-12))
        self.mse = nn.MSELoss()
        self.mi_loss = InfoNCE(self.hc_dim, self.hc_dim)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.classifier = nn.Linear(self.hc_dim, config.num_labels)

    def forward(self,
                original: Dict[str, Tensor],
                contrast: Dict[str, Tensor] = None,
                replace_index: Tensor = None,
                log_dict=None):
        loss = None
        if self.training:
            input_ids = torch.cat([original['input_ids'], contrast['input_ids']])
            token_type_ids = torch.cat([original['token_type_ids'], contrast['token_type_ids']])
            attention_mask = torch.cat([original['attention_mask'], contrast['attention_mask']])
            valid_mask = torch.cat([original['valid_mask'], contrast['valid_mask']])
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
            valid_input_ids = input_ids.view(-1)[valid_mask.view(-1) == 1].unsqueeze(-1)
            count: int = valid_mask.count_nonzero().item()
            tensor_cls = self.tokenizer.cls_token_id * torch.ones(
                count, 1, device=input_ids.device, dtype=torch.int64)
            tensor_sep = self.tokenizer.sep_token_id * torch.ones(
                count, 1, device=input_ids.device, dtype=torch.int64)
            concat_tensors = torch.cat([tensor_cls, valid_input_ids, tensor_sep], dim=-1)
            outputs = self.bert(concat_tensors)[0][:, 1]
            ht_loss = self.mse(self.ht_project(outputs),
                               ht.view(-1, self.ht_dim)[valid_mask.view(-1) == 1])
            # context-specific representation loss
            orig_hc, cont_hc = hc.chunk(2)
            replace_index_weights = (replace_index / replace_index.sum(-1, keepdim=True)).nan_to_num(0)
            hc_loss = self.mse(
                (orig_hc * replace_index_weights.unsqueeze(-1)).sum(1)[replace_index.sum(-1) != 0],
                (cont_hc * replace_index_weights.unsqueeze(-1)).sum(1)[replace_index.sum(-1) != 0],
            )
            # domain distribution loss
            info_loss = self.mi_loss.learning_loss(
                orig_hc.view(-1, self.hc_dim)[active_mask],
                cont_hc.view(-1, self.hc_dim)[active_mask],
            )
            jsd_loss = jsd(
                orig_logits.view(-1, self.num_labels)[original['valid_mask'].view(-1) == 1],
                cont_logits.view(-1, self.num_labels)[contrast['valid_mask'].view(-1) == 1])
            domain_loss = info_loss + jsd_loss
            log_dict({
                "ce_loss": ce_loss.item(),
                "orthogonal_loss": orthogonal_loss.item(),
                "reconstruct_loss": reconstruct_loss.item(),
                "replaced_token_loss": ht_loss.item(),
                "unreplaced_token_loss": hc_loss.item(),
                "info_loss": info_loss.item(),
                "jsd_loss": jsd_loss.item()
            })
            loss = ce_loss + 0.01 * orthogonal_loss + 0.1 * reconstruct_loss + hc_loss + ht_loss + self.beta * domain_loss
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