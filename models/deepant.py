import torch as t
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import BinaryPrecisionRecallCurve
from utils.eval import collect_metrics


class TCN(nn.Module):
    def __init__(self, feat_size: int, out_size: int, krn_size: int, pull_size):
        """
        A TCN module with causal 1D Convolution, ReLU activation and max-pooling.

        Args:
            feat_size: Dimension of input features
            out_size: Dimension of output features
            krn_size: Size of the convolution kernel
            pull_size: Size of the max-pull kernel
        """
        super().__init__()

        self.krn_size = krn_size
        self.conv = nn.Conv1d(
            in_channels=feat_size, out_channels=out_size,
            kernel_size=(krn_size,), padding='same'  # padding=(krn_size - 1) // 2
        )
        self.relu = nn.ReLU(inplace=True)
        self.max_pull = nn.MaxPool1d(kernel_size=pull_size)

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        # x = F.pad(x, (self.krn_size - 1, 0))  # causal
        x = self.conv(x)
        x = self.relu(x)
        x = self.max_pull(x)
        return x.transpose(1, 2)  # revert


class DeepAnT(L.LightningModule):
    def __init__(self, seq_dim: int, feat_dim: int, kernel_dim_1: int, kernel_dim_2: int, pull_dim: int, optimizer: str,
                 pred_horizon: int, lr: float):
        """
        The DeepAnT/TCN-P module.

        Args:
            seq_dim: Dimension of input sequences
            feat_dim: Dimension of input features
            kernel_dim_1: Size of the convolution kernel 1
            kernel_dim_2: Size of the convolution kernel 2
            pull_dim: Size of both max-pull kernels
            pred_horizon: Number of time steps to predict
        """
        super().__init__()

        self.lr = lr
        self.feat_dim = feat_dim
        self.pred_horizon = pred_horizon
        self.optimizer = optimizer
        self.results = {}
        self.preds = []
        self.trues = []
        self.block_1 = TCN(feat_dim, feat_dim, kernel_dim_1, pull_dim)
        self.block_2 = TCN(feat_dim, feat_dim, kernel_dim_2, pull_dim)
        self.fcl = nn.Linear(
            (seq_dim - pred_horizon) // (2 * pull_dim) * feat_dim,
            pred_horizon * feat_dim)


    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fcl(x)
        x_hat = x.view(x.shape[0], self.pred_horizon * self.feat_dim)
        return x_hat

    def _get_loss(self, x):
        x, y = (
            x[:, :-self.pred_horizon, :],
            x[:, -self.pred_horizon:, :]
        )
        y_hat = self(x)
        y = y.view(x.size(0), -1)
        fcst_loss = F.l1_loss(y_hat, y)
        return fcst_loss, y

    def training_step(self, batch, batch_idx):
        loss, _ = self._get_loss(batch)
        self.log('loss', loss, on_epoch=True, on_step=False)
        return loss

    def _refresh_step(self):
        self.trues = []
        self.preds = []

    def _eval_step(self, batch_):
        ref, tar = batch_
        x, y = (
            ref[:, :-self.pred_horizon, :],
            ref[:, -self.pred_horizon:, :]
        )
        y_hat = self(x)
        y = y.view(x.size(0), -1)
        pred_score = t.square(y_hat - y).view(y.size(0), -1).mean(1)
        self.preds.append(pred_score)
        loss = F.l1_loss(y_hat, y)
        y = (tar.sum(dim=1) > 0).int()
        self.trues.append(y)
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
