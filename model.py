import logging
from typing import Dict
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from torch import Tensor
from transformers.modeling_outputs import TokenClassifierOutput
from mi_estimators import InfoNCE
from mmd import MMD_loss

logger = logging.getLogger(__name__)


class FDGRPretrainedModel(BertPreTrainedModel):

    def __init__(self, config, h_dim: int):
        super(FDGRPretrainedModel, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.ha_dim = h_dim
        self.hc_dim = h_dim
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
        # self.mmd_loss = MMD_loss()
        self.mi_loss = InfoNCE(self.ha_dim, self.ha_dim)

    def forward(self,
                original: Dict[str, Tensor],
                word_contrast: Dict[str, Tensor] = None,
                replace_index: Tensor = None,
                context_contrast: Dict[str, Tensor] = None,
                original_index_select: Tensor = None,
                context_index_select: Tensor = None,
                log_dict=None):
        input_ids = torch.cat(
            [original['input_ids'], word_contrast['input_ids'], context_contrast["input_ids"]])
        token_type_ids = torch.cat([
            original['token_type_ids'], word_contrast['token_type_ids'],
            context_contrast["token_type_ids"]
        ])
        attention_mask = torch.cat([
            original['attention_mask'], word_contrast['attention_mask'],
            context_contrast["attention_mask"]
        ])
        seq_output: Tensor = self.bert(input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask)[0]
        ha, hc = self.ha_encoder(seq_output), self.hc_encoder(seq_output)
        orig_ha, word_cont_ha, context_cont_ha = ha.chunk(3)
        orig_hc, word_cont_hc, context_cont_hc = hc.chunk(3)
        # orthogonal loss
        orthogonal_loss = torch.matmul(
            ha.view(-1, self.ha_dim)[attention_mask.view(-1) == 1],
            hc.view(-1, self.hc_dim)[attention_mask.view(-1) == 1].T).abs().mean()
        # auto-encoder loss
        reconstruct_loss = self.mse(
            seq_output.view(-1, seq_output.size(-1))[attention_mask.view(-1) == 1],
            self.decoder(torch.cat([ha, hc],
                                   dim=-1)).view(-1,
                                                 seq_output.size(-1))[attention_mask.view(-1) == 1])
        # attribute-specific representation loss
        active_mask = original['attention_mask'].view(-1) == 1
        ha_loss = self.mi_loss.learning_loss(
            orig_ha.view(-1, self.ha_dim)[active_mask],
            word_cont_ha.view(-1, self.ha_dim)[active_mask])
        # contenc-specific representation loss
        batch_size = original["input_ids"].size(0)
        t1, t2 = [], []
        for i in range(batch_size):
            s1, s2 = original_index_select[i], context_index_select[i]
            t1.append(orig_hc[i].index_select(0, s1[s1 != -1]))
            t2.append(context_cont_hc[i].index_select(0, s2[s2 != -1]))
        hc_loss = self.mi_loss.learning_loss(torch.cat(t1), torch.cat(t2))
        if log_dict:
            log_dict({
                "orthogonal_loss": orthogonal_loss.item(),
                "reconstruct_loss": reconstruct_loss.item(),
                "ha_loss": ha_loss.item(),
                "hc_loss": hc_loss.item(),
            })
        loss = 0.1 * orthogonal_loss + reconstruct_loss + ha_loss + hc_loss
        return TokenClassifierOutput(loss=loss, hidden_states=(ha, hc))


class FDGRModel(nn.Module):

    def __init__(self, num_labels: int, pretrained_path: str):
        super(FDGRModel, self).__init__()
        weights = torch.load(pretrained_path)
        pretrained_model_name = weights["hyper_parameters"]["pretrained_model_name"]
        h_dim = weights["hyper_parameters"]["h_dim"]
        self.fdgr = FDGRPretrainedModel.from_pretrained(pretrained_model_name, h_dim)
        self.fdgr.load_state_dict(
            {k.replace("model.", ''): v
             for k, v in weights['state_dict'].items()})
        self.num_labels = num_labels
        # self.att = nn.MultiheadAttention(h_dim, 2, batch_first=True)
        # self.mi_loss = InfoNCE(self.hc_dim, self.hc_dim)
        # self.club_loss = vCLUB()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.classifier = nn.Linear(h_dim, num_labels)
        # self.mmd_loss = MMD_loss()
        # self.pos_emb = nn.Embedding(len(POS_DICT), self.pos_dim, POS_DICT.get(self.tokenizer.pad_token))
        # self.domain_classifier = nn.Sequential(RevGrad(), nn.Linear(config.hidden_size, 1))
        # self.domain_loss = nn.BCEWithLogitsLoss()

    def forward(self,
                original: Dict[str, Tensor],
                word_contrast: Dict[str, Tensor] = None,
                replace_index: Tensor = None,
                context_contrast: Dict[str, Tensor] = None,
                original_index_select: Tensor = None,
                context_index_select: Tensor = None,
                log_dict=None):
        outputs: TokenClassifierOutput = self.fdgr(original, word_contrast, replace_index,
                                                   context_contrast, original_index_select,
                                                   context_index_select, log_dict)
        ha, hc = outputs.hidden_states
        original_ha = ha.chunk(3)[0]
        logits: Tensor = self.classifier(original_ha)
        active_mask = original['valid_mask'].view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_mask]
        ce_loss = self.loss_fct(logits.view(-1, self.num_labels), original['gold_labels'].view(-1))
        loss = ce_loss + 0.01 * outputs.loss
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