import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from utils.eval import collect_metrics


class Encoder(nn.Module):
  def __init__(self, feat_dim: int, lat_dim_: int, num_layer_: int, dropout_: float):
    super().__init__()

    self.num_layer_ = num_layer_
    self.lat_dim_ = lat_dim_
    self.lstm = nn.LSTM(
      input_size=feat_dim, hidden_size=lat_dim_,
      batch_first=True, num_layers=num_layer_,
      bidirectional=False, dropout=dropout_
    )

  def forward(self, x):
    hidden = self._get_states(x.size(0))
    hidden = (hidden[0].to(x), hidden[1].to(x))
    z, hidden = self.lstm(x, hidden)
    return z, hidden

  def _get_states(self, batch_size):
    return t.randn(self.num_layer_, batch_size, self.lat_dim_), \
    t.randn(self.num_layer_, batch_size, self.lat_dim_)


class Decoder(nn.Module):
  def __init__(self, feat_dim: int, lat_dim_: int, num_layer_: int, dropout_: float):
    super().__init__()

    self.num_layer_ = num_layer_
    self.lat_dim_ = lat_dim_
    self.lstm = nn.LSTM(
        input_size=lat_dim_, hidden_size=lat_dim_,
        batch_first=True, num_layers=num_layer_,
        bidirectional=False, dropout=dropout_
    )
    self.fc = nn.Sequential(
      nn.Linear(lat_dim_, feat_dim), nn.Sigmoid()
    )

  def forward(self, z, hidden):
    z, hidden = self.lstm(z, hidden)
    y = self.fc(z)
    return y


class LSTMAE(L.LightningModule):
  def __init__(self, seq_dim:int, feat_dim: int, lat_dim: int, num_layer: int, dropout: float, pooling: str, lr: float, optimizer: str,
               encoder: nn.Module = Encoder, decoder: nn.Module = Decoder):
    super().__init__()

    self.lr = lr
    self.best_thresh = None
    self.optimizer = optimizer
    self.results = {}
    self.val_res = []
    self.trues = []
    self.preds = []
    self.pooling = pooling
    self.encoder = encoder(
    feat_dim, lat_dim, num_layer, dropout
    )
    self.decoder = decoder(
    feat_dim, lat_dim, num_layer, dropout
    )

  def forward(self, x):
    lat, state = self.encoder(x)
    lat = self.pool(lat)
    lat = lat.unsqueeze(1).repeat(1, x.size(1), 1)
    y = self.decoder(lat, state)
    return y

  def pool(self, batch):
    if self.pooling == 'max':
        pooled = batch.max(dim=1)[0]
    elif self.pooling == 'mean':
        pooled = batch.mean(dim=1)
    elif self.pooling == 'last':
        pooled = batch[:, -1, :]
    else:
        raise ValueError(f'Unknown pooling: {self.pooling}')
    return pooled

  def training_step(self, batch, batch_idx):
    rec = self.forward(batch)
    loss = F.mse_loss(rec, batch)
    self.log('loss', loss, on_epoch=True, on_step=False)
    return loss

  def _eval_step(self, batch):
    x, y = batch
    rec = self.forward(x)
    pred_score = t.square(x - rec).view(x.size(0), -1).mean(1)
    self.preds.append(pred_score)
    y = (y.sum(dim=1) > 0).int()
    self.trues.append(y)
    loss = F.mse_loss(rec, x)
    return loss

  def validation_step(self, batch, batch_idx):
      val_loss = self._eval_step(batch)
      self.log('test-loss', val_loss, on_epoch=True, on_step=False)

  def test_step(self, batch, batch_idx):
      test_loss = self._eval_step(batch)
      self.log('test-loss', test_loss, on_epoch=True, on_step=False)

  def _refresh_step(self):
      self.trues = []
      self.preds = []

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
