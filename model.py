import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv
from torch import Tensor
from transformers import BertModel, BertConfig
from constants import DEPREL_DICT, POS_DICT

from mi_estimators import InfoNCE, vCLUB
from span_extractors import EndpointSpanExtractor

logger = logging.getLogger(__name__)


@dataclass
class TokenClassifierOutput():
    loss: Optional[Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class FDGRPretrainedModel(nn.Module):

    def __init__(self, config: BertConfig, h_dim: int):
        super(FDGRPretrainedModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.name_or_path)
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
        stacked = torch.stack([v, a, d], -2)  # [batch_size, seq_len, 3, h_dim]
        I = torch.eye(3, device=stacked.device)
        orthogonal_loss = torch.norm(torch.matmul(stacked, stacked.transpose(-1, -2)) - I)
        return TokenClassifierOutput(loss={
            "club_loss": club_loss,
            "reconstruct_loss": reconstruct_loss,
            "orthogonal_loss": orthogonal_loss,
            "ha_loss": ha_loss,
            "vad_loss": vad_loss
        },
                                     hidden_states=(ha, hc))


class FDGRModel(nn.Module):

    def __init__(self, config: BertConfig, h_dim: int):
        super(FDGRModel, self).__init__()
        self.h_dim = h_dim
        self.fdgr = FDGRPretrainedModel(config, h_dim)
        self.num_labels: int = config.num_labels
        self.pos_embedding = nn.Embedding(len(POS_DICT), 30, 0)
        self.dep_embedding = nn.Embedding(len(DEPREL_DICT), 30, 0)
        self.conv1 = GraphConv(h_dim, h_dim, allow_zero_in_degree=True)
        # self.conv2 = GraphConv(h_dim, h_dim, allow_zero_in_degree=True)
        self.span_extractor = EndpointSpanExtractor(h_dim,
                                                    combination="x,y",
                                                    num_width_embeddings=4,
                                                    span_width_embedding_dim=50,
                                                    bucket_widths=True)
        self.classifier = nn.Linear(3 * h_dim + 50, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, original: Dict[str, Tensor], contrast: Dict[str, Tensor] = None):
        outputs: TokenClassifierOutput = self.fdgr(original, contrast)
        ha, hc = outputs.hidden_states
        if self.training:
            att_mask = torch.cat([original['attention_mask'], contrast['attention_mask']])
            valid_mask = torch.cat([original['valid_mask'], contrast['valid_mask']])
            pos_ids = torch.cat([original["pos_ids"], contrast["pos_ids"]])
            dep_ids = torch.cat([original["dep_ids"], contrast["dep_ids"]])
            graph_ids = torch.cat([original['graph_ids'], contrast['graph_ids']])
            aspect_ids = torch.cat([original['aspect_ids'], contrast['aspect_ids']])
            span_indices = torch.cat([original['span_indices'], contrast['span_indices']])
        else:
            att_mask = original['attention_mask']
            valid_mask = original['valid_mask']
            pos_ids = original["pos_ids"]
            dep_ids = original["dep_ids"]
            graph_ids = original['graph_ids']
            aspect_ids = original['aspect_ids']
            span_indices = original['span_indices']
        batch_size = pos_ids.size(0)
        pos = self.pos_embedding(pos_ids)
        dep = self.dep_embedding(dep_ids)
        g = dgl.batch([
            dgl.graph(torch.nonzero(adj_mat, as_tuple=True), num_nodes=adj_mat.size(-1))
            for adj_mat in graph_ids
        ])
        h = self.conv1(g, ha.view(-1, self.h_dim))
        # h = self.conv2(g, h.view(-1, self.h_dim))
        # sequence representetion
        mean = torch.mul(h.view(batch_size, -1, self.h_dim),
                         (att_mask / att_mask.sum(-1, True)).unsqueeze(-1)).sum(1)
        # emb = torch.mul(h.view(batch_size, -1, self.h_dim),
        #                 (valid_mask / valid_mask.sum(-1, True)).unsqueeze(-1)).sum(1)
        spans = self.span_extractor(h.view(batch_size, -1, self.h_dim), span_indices,
                                    att_mask).squeeze(1)
        emb = torch.cat([spans, mean], dim=-1)
        logits: Tensor = self.classifier(emb)
        if self.training:
            outputs.loss["ce_loss"] = self.loss_fct(
                logits, torch.cat([original['gold_labels'], contrast["gold_labels"]]))
        return TokenClassifierOutput(logits=logits, loss=outputs.loss, hidden_states=emb)


class BertForTokenClassification(nn.Module):

    def __init__(self, pretrained_model_name, num_labels):
        super(BertForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(pretrained_model_name, num_labels=num_labels)
        config = BertConfig.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self,
                input_ids: Tensor,
                token_type_ids: Tensor = None,
                attention_mask: Tensor = None,
                valid_mask: Tensor = None,
                gold_labels: Tensor = None,
                aspect_ids: Tensor = None,
                span_indices: Tensor = None,
                pos_ids: Tensor = None,
                dep_ids: Tensor = None,
                graph_ids: Tensor = None,
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
