import logging
from typing import Dict
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torch import Tensor
from transformers.modeling_outputs import TokenClassifierOutput
from mi_estimators import InfoNCE, CLUBMean, vCLUB, jsd

logger = logging.getLogger(__name__)


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
                labeled: Tensor = None,
                log_dict=None,
                batch_rate: int = -1):
        input_ids = torch.cat([original['input_ids'], word_contrast['input_ids']])
        token_type_ids = torch.cat([original['token_type_ids'], word_contrast['token_type_ids']])
        attention_mask = torch.cat([original['attention_mask'], word_contrast['attention_mask']])
        seq_output: Tensor = self.bert(input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask)[0]
        orig_seq_output, word_cont_seq_output = seq_output.chunk(2)
        ha = self.ha_encoder(seq_output)
        hc = self.hc_encoder(seq_output)
        orig_ha, word_cont_ha = ha.chunk(2)
        orig_hc, word_cont_hc = hc.chunk(2)
        # orthogonal loss
        orthogonal_loss = self.orthogonal_loss.update(
            ha.view(-1, self.ha_dim)[attention_mask.view(-1) == 1],
            hc.view(-1, self.hc_dim)[attention_mask.view(-1) == 1])
        # auto-encoder loss
        decoded = self.decoder(torch.cat([ha, hc], dim=-1))
        reconstruct_loss = self.mse(
            seq_output.view(-1, self.hidden_size)[attention_mask.view(-1) == 1],
            decoded.view(-1, self.hidden_size)[attention_mask.view(-1) == 1])
        # attribute-specific representation loss
        assert torch.all(original['attention_mask'] == word_contrast['attention_mask']).item() == 1
        active_mask = original['attention_mask'].view(-1) == 1
        ha_loss = self.mi_loss.learning_loss(
            orig_ha.view(-1, self.ha_dim)[active_mask],
            word_cont_ha.view(-1, self.ha_dim)[active_mask])
        cross_mi_loss = self.cross_mi_loss.learning_loss(
            orig_seq_output.view(-1, self.hidden_size)[active_mask],
            word_cont_ha.view(-1, self.ha_dim)[active_mask])
        cross_mi_loss += self.cross_mi_loss.learning_loss(
            word_cont_seq_output.view(-1, self.hidden_size)[active_mask],
            orig_ha.view(-1, self.ha_dim)[active_mask])
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
        if log_dict:
            log_dict({
                "orthogonal_loss": orthogonal_loss.item(),
                "reconstruct_loss": reconstruct_loss.item(),
                "ha_loss": ha_loss.item(),
                "cross_mi_loss": cross_mi_loss.item(),
                "hc_loss_replaced": hc_loss_replaced.item(),
                "hc_loss_unreplaced": hc_loss_unreplaced.item(),
            })
        loss = self.weight1(batch_rate) * orthogonal_loss + reconstruct_loss + ha_loss + cross_mi_loss + \
            self.weight2(batch_rate) * hc_loss_replaced + hc_loss_unreplaced
        return TokenClassifierOutput(loss=loss, hidden_states=(ha, hc))

    def weight1(self, batch_rate: float):
        return 0.1 * torch.sigmoid(torch.as_tensor(20 * (batch_rate - 1.0 / 3)))

    def weight2(self, batch_rate: float) -> Tensor:
        return 0.1 * torch.sigmoid(torch.as_tensor(20 * (batch_rate - 1.0 / 3)))


class FDGRModel(nn.Module):

    def __init__(self, num_labels: int, pretrained_path: str):
        super(FDGRModel, self).__init__()
        weights = torch.load(pretrained_path)
        pretrained_model_name = weights["hyper_parameters"]["pretrained_model_name"]
        h_dim = weights["hyper_parameters"]["h_dim"]
        self.fdgr = FDGRPretrainedModel.from_pretrained(pretrained_model_name, h_dim)
        # self.fdgr.load_state_dict(
        #     {k.replace("model.", ''): v
        #      for k, v in weights['state_dict'].items()})
        self.num_labels = num_labels
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.classifier = nn.Linear(h_dim, num_labels)
        self.mi_loss = InfoNCE(h_dim, num_labels)

    def forward(self,
                original: Dict[str, Tensor],
                word_contrast: Dict[str, Tensor] = None,
                replace_index: Tensor = None,
                labeled: Tensor = None,
                log_dict=None,
                batch_rate: int = -1):
        outputs: TokenClassifierOutput = self.fdgr(original, word_contrast, replace_index, labeled, log_dict,
                                                   batch_rate)
        ha, hc = outputs.hidden_states
        logits: Tensor = self.classifier(ha.chunk(2)[0]) / 2
        active_mask = original['valid_mask'].view(-1) == 1
        label_mask = labeled.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_mask]
        ce_loss = self.loss_fct(logits.view(-1, self.num_labels)[label_mask], 
                                original['gold_labels'].view(-1)[label_mask])
        # loss = ce_loss + 0.01 * outputs.loss
        loss = ce_loss + 0.01 * self.mi_loss.learning_loss(ha.chunk(2)[0].view(-1, ha.size(-1))[active_mask], active_logits)
        return TokenClassifierOutput(logits=active_logits, loss=loss)


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