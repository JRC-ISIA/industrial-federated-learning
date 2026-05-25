import torch as t
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse, remove_self_loops

from utils.eval import collect_metrics


class Conv1d(nn.Module):
    def __init__(self, feat_size: int, krn_size: int):
        """
        1D Convolutional module

        Args:
            feat_size: Dimension of input features
            krn_size: Size of the convolving kernel
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=feat_size, out_channels=feat_size,
            kernel_size=(krn_size,), padding=(krn_size - 1) // 2
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class GAT(nn.Module):
    def __init__(self, num_nodes: int, node_size: int, batch_size: int):
        super().__init__()

        self.num_nodes = num_nodes

        # Pre-calculate for the largest possible batch once
        adj = t.ones(num_nodes, num_nodes) # fully connected including self-loops
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = remove_self_loops(edge_index)

        # Vectorized Batching for the max size
        offsets = t.arange(batch_size).view(-1, 1) * num_nodes
        batched_edge_index = (edge_index.unsqueeze(0) + offsets.unsqueeze(1)).transpose(0, 1).reshape(2, -1)
        self.register_buffer('full_edge_index', batched_edge_index, persistent=False)

        self.spatial_gat = GATConv(
          in_channels=node_size,
          out_channels= node_size,
          add_self_loops=True,  # False if adj. matrix already has self-loops
          # heads = 3,  # multihead attention
          # concat=False
          )

    def forward(self, x):
        B, N, H = x.shape
        # Dynamic Slicing: Only use the portion of the buffer needed for the current batch
        # Number of edges per graph is N*(N-1) if fully connected without self-loops
        edges_per_graph = self.num_nodes * (self.num_nodes - 1)
        current_edge_index = self.full_edge_index[:, :B * edges_per_graph]

        x_flat = x.reshape(-1, H)
        out = self.spatial_gat(x_flat, current_edge_index)
        return out.view(B, N, -1)


class GRU(nn.Module):
    def __init__(self, feat_dim: int, lat_dim: int, dropout: float = 0.0):
        super().__init__()

        self.gru = nn.GRU(
            feat_dim, lat_dim, batch_first=True, dropout=dropout
        )

    def forward(self, x):
        out, h = self.gru(x)
        return out, h


class VAE(nn.Module):  # details of vae is not clear in the paper
    def __init__(self, lat_dim: int, feat_dim: int):
        super().__init__()

        self.decoder = GRU(lat_dim, feat_dim)
        self.z_mu = nn.Linear(lat_dim, lat_dim)
        self.z_log_var = nn.Linear(lat_dim, lat_dim)
        self.x_hat_mu = nn.Linear(feat_dim, feat_dim)
        self.x_hat_logvar = nn.Linear(feat_dim, feat_dim)

    def forward(self, x, x_):
        z = x[:, -1, :] 
        x_ = x_[:, -1, :]  # reference
        z_mu, z_log_var = self.z_mu(z), self.z_log_var(z)
        # z_var = F.softplus(z_log_var)
        z = self._reparam_gauss(z_mu, z_log_var)

        x_hat, _ = self.decoder(z)
        x_hat_mu, x_hat_logvar = self.x_hat_mu(x_hat), self.x_hat_logvar(x_hat)
        x_var = F.softplus(x_hat_logvar)
        kld_loss = t.mean(
            -0.5 * t.sum(1 + z_log_var - z_mu ** 2 - z_log_var.exp(), dim=-1)
        )
        # rec_loss = F.mse_loss(x_hat, x_)
        nll = F.gaussian_nll_loss(
            x_hat_mu, x_, x_var, reduction="none"
        ).sum(dim=-1).mean()  # comparable scales with kld_loss sum >> mean
        loss = kld_loss + nll
        return loss, x_hat_mu, x_var

    def _reparam_gauss(self, mu, log_var):
        std = log_var.mul(0.5).exp_()#.to(mu)  # Tensor.to for device conversion, scales with number of gpus/tpus.
        eps = t.randn_like(std, dtype=mu.dtype)  # gaussian distribution, typical choice
        lat = eps.mul(std).add_(mu)  # z = μ + σ * ϵ, with ϵ ~ N(0,I)
        return lat


class MLP(nn.Module):
    def __init__(self, embed_dim: int, lat_dim: int, feat_dim: int):
        super().__init__()

        self.fcn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, lat_dim), nn.ReLU(),
            nn.Linear(lat_dim, lat_dim), nn.ReLU(),
            nn.Linear(lat_dim, feat_dim),
        )

    def forward(self, x, x_):
        x, y = x[:, :-1, :], x[:, -1, :]
        y_hat = self.fcn(x)
        y = x_[:, -1, :]
        return y, y_hat


class MTADGAT(L.LightningModule):
    def __init__(self, seq_len: int, feat_len: int, embed_dim: int, embed_dim_2:int, conv_dim: int, batch_size:int, gamma: float, lr: float, optimizer: str):
        super().__init__()

        self.lr = lr
        self.gamma = gamma
        self.optimizer = optimizer
        self.preds = []
        self.trues = []
        self.results = {}
        self.conv_1d = Conv1d(feat_len, conv_dim)
        self.feat_gat = GAT(feat_len, seq_len, 0)
        self.temp_gat = GAT(seq_len, feat_len, 0)
        self.gru = GRU(3 * feat_len, embed_dim)
        self.vae = VAE(embed_dim, feat_len)
        self.mlp = MLP((seq_len - 1) * embed_dim, embed_dim_2, feat_len)

    def forward(self, x):
        conv = self.conv_1d(x.transpose(1, 2))
        feat_gat = self.feat_gat(conv)
        feat_gat = feat_gat.transpose(1, 2)
        conv = conv.transpose(1, 2)
        temp_gat = self.temp_gat(conv)
        embedding = t.concat([conv, temp_gat, feat_gat], dim=-1)
        gru_out, _ = self.gru(embedding)
        rec_loss, rec_mu, rec_var = self.vae(gru_out, x)
        target, forecast = self.mlp(gru_out, x)
        return rec_loss, target, forecast, rec_mu, rec_var

    def _get_loss(self, x):
        rec_loss, y, y_hat, _, _ = self(x)
        fcst_loss = F.mse_loss(y_hat, y, reduction='none').sum(-1).mean().sqrt()  # RMSE
        return fcst_loss + rec_loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('loss', loss, on_epoch=True, on_step=False)
        return loss

    def _refresh_step(self):
        self.trues = []
        self.preds = []

    def _eval_step(self, batch):
        x, y = batch
        rec_loss, y_, y_hat, x_hat_mu, x_hat_var = self(x)

        x = x[:, -1, :]  # targeting common (between reconstruction and forecast) point

        nll = F.gaussian_nll_loss(x_hat_mu, x, x_hat_var, reduction='none', full=True)
        t.mul(nll, -1, out=nll)
        t.expm1(nll, out=nll)
        t.mul(nll, -1, out=nll)
        anomaly_score = t.sum(nll, dim=-1)
        # t.nan_to_num(anomaly_score, out=anomaly_score)

        pred_score = t.sum((y_hat - y_) ** 2, dim=-1)
        pred_score += (self.gamma * anomaly_score)
        pred_score /= (1 + self.gamma)
        self.preds.append(pred_score)

        loss = rec_loss + F.mse_loss(y_hat, y_, reduction='none').sum(-1).mean().sqrt()
        self.log('test-loss', loss, on_epoch=True, on_step=False)
        y = (y.sum(dim=1) > 0).int()
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
