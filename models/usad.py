import torch as t
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import BinaryPrecisionRecallCurve
from utils.eval import collect_metrics


class Encoder(nn.Module):
    def __init__(self, seq_dim_: int, lat_dim_: int):
        """
        USAD encoder model

        Args:
            seq_dim_: Dimension of input sequence
            lat_dim_: Dimension of latent space
        """
        super().__init__()

        self.encode = nn.Sequential(
            nn.Linear(seq_dim_, seq_dim_ // 2), nn.ReLU(True),
            nn.Linear(seq_dim_ // 2, seq_dim_ // 4), nn.ReLU(True),
            nn.Linear(seq_dim_ // 4, lat_dim_), nn.ReLU(True)
        )

    def forward(self, w):
        z = self.encode(w)
        return z


class Decoder(nn.Module):
    def __init__(self, lat_dim_: int, seq_dim_: int):
        """
        USAD decoder model

        Args:
            lat_dim_: Dimension of latent space
            seq_dim_: Dimension of input sequence
        """
        super().__init__()

        self.decode = nn.Sequential(
            nn.Linear(lat_dim_, seq_dim_ // 4), nn.ReLU(True),
            nn.Linear(seq_dim_ // 4, seq_dim_ // 2), nn.ReLU(True),
            nn.Linear(seq_dim_ // 2, seq_dim_), nn.Sigmoid()
        )

    def forward(self, z):
        w = self.decode(z)
        return w


class USAD(L.LightningModule):
    def __init__(self, seq_dim: int, feat_dim: int, lat_dim: int, lr: float, optimizer: str,
                 encoder: nn.Module = Encoder, decoder: nn.Module = Decoder):
        """
        USAD autoencoder submodules

        Args:
            seq_dim: Dimension  of input sequence
            feat_dim: Dimension of input features
            lat_dim: Dimension of latent space
            encoder: Instance of a module for encoding
            decoder: Instance of a module for decoding
        """
        super().__init__()

        self.lr = lr
        self.best_thresh = None
        self.optimizer = optimizer
        self.results = {}
        self.preds =[]
        self.trues = []
        in_dim = seq_dim * feat_dim
        self.encoder = encoder(in_dim, lat_dim)
        self.decoder_1 = decoder(lat_dim, in_dim)
        self.decoder_2 = decoder(lat_dim, in_dim)
        self.automatic_optimization = False


    def forward(self, x):
        z_1 = self.encoder(x)
        x_1 = self.decoder_1(z_1)
        x_2 = self.decoder_2(z_1)
        z_2 = self.encoder(x_1)
        x_3 = self.decoder_2(z_2)
        return x_1, x_2, x_3

    def _get_loss(self, x):
        w_1, w_2, w_3 = self(x)
        n = self.current_epoch + 1
        l_1 = (1 / n) * F.mse_loss(x, w_1) + (1 - 1 / n) * F.mse_loss(x, w_3)
        l_2 = (1 / n) * F.mse_loss(x, w_2) - (1 - 1 / n) * F.mse_loss(x, w_3)
        return l_1, l_2

    def training_step(self, batch, batch_idx):
        opt_1, opt_2 = self.optimizers()

        self.toggle_optimizer(opt_1)
        loss_1, loss_2 = self._get_loss(batch)
        opt_1.zero_grad()
        self.manual_backward(loss_1)
        opt_1.step()
        self.untoggle_optimizer(opt_1)

        self.toggle_optimizer(opt_2)
        loss_1, loss_2 = self._get_loss(batch)
        opt_2.zero_grad()
        self.manual_backward(loss_2)
        opt_2.step()
        self.untoggle_optimizer(opt_2)
        self.log('loss', loss_2, on_epoch=True, on_step=False)

    def _refresh_step(self):
        self.trues = []
        self.preds = []

    def _eval_step(self, batch, alpha=.5, beta=.5):
        x, y = batch
        w_1 = self.decoder_1(self.encoder(x))
        w_2 = self.decoder_2(self.encoder(w_1))
        pred_score = alpha * t.mean((x - w_1) ** 2, dim=1) + beta * t.mean((x - w_2) ** 2, dim=1)
        self.preds.append(pred_score)
        y = (y.sum(dim=1) > 0).int()
        self.trues.append(y)
        loss = F.mse_loss(w_2, x)
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
        opt_1 = getattr(t.optim, self.optimizer)(list(self.encoder.parameters()) + list(self.decoder_1.parameters()), lr=self.lr)
        opt_2 = getattr(t.optim, self.optimizer)(list(self.encoder.parameters()) + list(self.decoder_2.parameters()), lr=self.lr)
        return opt_1, opt_2
