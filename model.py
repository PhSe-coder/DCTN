import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel, BertPreTrainedModel
from torch import Tensor
from transformers.modeling_outputs import TokenClassifierOutput
from mi_estimators import InfoNCE
import torch.utils.checkpoint as cp

logger = logging.getLogger(__name__)


class MIBert(BertPreTrainedModel):

    def __init__(self, config, alpha, tau):
        super(MIBert, self).__init__(config)
        self.bert = BertModel(config)
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("tau", torch.tensor(tau))
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(0.1)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        # self.mi_loss = MILoss()
        self.mi_loss = InfoNCE(config.hidden_size, config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                batch_tgt: Dict[str, Tensor],
                batch_src: Dict[str, Tensor] = None,
                student: bool = True):
        loss, ce_loss, _mi_loss = None, None, 0
        if self.training:
            input_ids = torch.cat([batch_src['input_ids'], batch_tgt['input_ids']])
            token_type_ids = torch.cat([batch_src['token_type_ids'], batch_tgt['token_type_ids']])
            attention_mask = torch.cat([batch_src['attention_mask'], batch_tgt['attention_mask']])
            # valid_mask = torch.cat([batch_src['valid_mask'], batch_tgt['valid_mask']])
            sequence_output = self.bert(input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)[0]
            sequence_output = self.dropout(sequence_output)
            logits: Tensor = self.classifier(sequence_output) / self.tau
            src_logits, tgt_logits = torch.chunk(logits, 2)
            _, tgt_outputs = torch.chunk(sequence_output, 2)
            active_mask = batch_tgt['attention_mask'].view(-1) == 1
            active_tgt_logits = tgt_logits.view(-1, self.num_labels)[active_mask]
            hidden_states = tgt_outputs.view(-1, tgt_outputs.size(-1))[active_mask]
            if student:
                gold_labels = batch_src['gold_labels']
                ce_loss = self.loss_fct(src_logits.view(-1, src_logits.size(-1)),
                                        gold_labels.view(-1))
                active_logits = logits.view(-1, logits.size(-1))[attention_mask.view(-1) == 1]
                # use checkpoint to avoid cuda out of memory although making the running slower.
                _mi_loss = cp.checkpoint(
                    self.mi_loss.learning_loss,
                    sequence_output.view(-1,
                                         sequence_output.size(-1))[attention_mask.view(-1) == 1],
                    f.softmax(active_logits, dim=-1))
                # p = f.softmax(active_logits, dim=-1)
                # log_p = f.log_softmax(active_logits, dim=-1)
                # _mi_loss = self.mi_loss(p, log_p)
                # _mi_loss = self.mi_loss.learning_loss(
                #     sequence_output.view(-1, sequence_output.size(-1))[valid_mask.view(-1) == 1],
                #     f.softmax(active_logits, dim=-1))
                loss = ce_loss + self.alpha * _mi_loss
        else:
            input_ids = batch_tgt['input_ids']
            attention_mask = batch_tgt['attention_mask']
            token_type_ids = batch_tgt['token_type_ids']
            valid_mask = batch_tgt['valid_mask']
            sequence_output = self.bert(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)[0]
            active_mask = valid_mask.view(-1) == 1
            tgt_logits: Tensor = self.classifier(sequence_output)
            active_tgt_logits = tgt_logits.view(-1, self.num_labels)[active_mask]
            hidden_states = sequence_output.view(-1, sequence_output.size(-1))[active_mask]
        return TokenClassifierOutput(logits=active_tgt_logits,
                                     loss=(loss, self.alpha * _mi_loss, ce_loss),
                                     hidden_states=hidden_states)


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