import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
import lightning as L
from utils.eval import collect_metrics


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=.1):
        super().__init__()

        pe = t.zeros(max_len, d_model, dtype=t.float)
        pos = t.arange(0, max_len, dtype=t.float).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, dtype=t.float) * (-math.log(10000.0) / d_model))
        pe += t.sin(pos * div_term)
        pe += t.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dim_feedforward: int, dropout: float):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout
        )
        self.fcn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.LeakyReLU(True),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.drop1(src2)
        src2 = self.fcn(src)
        src = src + self.drop2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dim_feedforward: int, dropout: float):
        super().__init__()

        self.dropout = dropout
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout
        )
        self.multi_head_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout
        )
        self.fcn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.LeakyReLU(True),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + F.dropout(tgt2, self.dropout)
        tgt2 = self.multi_head_attn(tgt, memory, memory)[0]
        tgt = tgt + F.dropout(tgt2, self.dropout)
        tgt2 = self.fcn(tgt)
        tgt = tgt + F.dropout(tgt2, self.dropout)
        return tgt


# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(L.LightningModule):
    def __init__(self, seq_dim: int, feat_dim: int, lat_dim:int, dropout: float, lr: float, w_decay: float, optimizer: str):
        super().__init__()

        self.lr = lr
        self.feat_dim = feat_dim
        self.w_decay = w_decay
        self.optimizer = optimizer
        self.results = {}
        self.trues = []
        self.preds = []
        self.pos_encoder = PositionalEmbedding(2 * feat_dim, seq_dim, .1)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=2 * feat_dim, n_head=feat_dim, dim_feedforward=lat_dim, dropout=dropout),
            enable_nested_tensor=True, num_layers=1)
        self.transformer_decoder1 = TransformerDecoder(
            TransformerDecoderLayer(d_model=2 * feat_dim, n_head=feat_dim, dim_feedforward=lat_dim, dropout=dropout),
            num_layers=1
        )
        self.transformer_decoder2 = TransformerDecoder(
            TransformerDecoderLayer(d_model=2 * feat_dim, n_head=feat_dim, dim_feedforward=lat_dim, dropout=dropout),
            num_layers=1
        )
        self.fcn = nn.Sequential(nn.Linear(2 * feat_dim, feat_dim), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = t.cat((src, c), dim=2)
        src = src * math.sqrt(self.feat_dim)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = t.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

    def _get_loss(self, x, train=False):
        batch_size, features = x.shape[0], x.shape[2]
        x = x.permute(1, 0, 2)  # (batch, seq, feat) -> (seq, batch, feat)
        elem = x[-1, :, :].view(1, batch_size, features)
        embed_1, embed_2 = self(x, elem)
        n = self.current_epoch + 1
        loss_1 = F.mse_loss(embed_1, elem, reduction='none')
        loss_2 = F.mse_loss(embed_2, elem, reduction='none')
        if train:
            loss = (1 / n) * loss_1.mean() + (1 - 1 / n) * loss_2.mean()
        else:
            loss = (loss_2.mean(dim=-1).squeeze(0), loss_2.mean())
        return loss

    def training_step(self, batch, batch_index):
        loss = self._get_loss(batch, train=True)
        self.log('loss', loss, on_epoch=True, on_step=False)
        return loss

    def _refresh_step(self):
        self.trues = []
        self.preds = []

    def _eval_step(self, batch):
        x, y = batch
        pred_score = self._get_loss(x, train=False)[0]
        self.preds.append(pred_score)
        y = (y.sum(dim=1) > 0).int()
        self.trues.append(y)
        loss = pred_score.mean()
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self._eval_step(batch)
        self.log('test-loss', val_loss, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        test_loss = self._eval_step(batch)
        self.log('test-loss', test_loss, on_epoch=True, on_step=False)

    def on_validation_epoch_start(self):
        self._refresh_step()

    def on_test_epoch_start(self):
        self._refresh_step()

    def _get_metrics(self):
        vus_roc, vus_pr, auc_roc, auc_pr, best_f1_p, best_f1_c = collect_metrics(
          t.cat(self.trues).detach(),
          t.cat(self.preds).detach(),
        )
        self.results['VUS_ROC'] = vus_roc
        self.results['VUS_PR'] = vus_pr
        self.results['AUC_ROC'] = auc_roc
        self.results['AUC_PR'] = auc_pr
        self.results['Best-F1-Point-Wise'] = best_f1_p
        self.results['Best-F1-Composite'] = best_f1_c

        self.log('VUS_ROC', vus_roc, on_epoch=True, on_step=False)
        self.log('VUS_PR', vus_pr, on_epoch=True, on_step=False)
        self.log('AUC_ROC', auc_roc, on_epoch=True, on_step=False)
        self.log('AUC_PR', auc_pr, on_epoch=True, on_step=False)
        self.log('Best-F1-Point-Wise', best_f1_p, on_epoch=True, on_step=False)
        self.log('Best-F1-Composite', best_f1_c, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        self._get_metrics()

    def on_test_epoch_end(self):
        self._get_metrics()

    def configure_optimizers(self):
        optimizer = getattr(t.optim, self.optimizer)(self.parameters(), lr=self.lr)
        return optimizer
