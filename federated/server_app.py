from typing import Dict, Optional, Tuple
import logging
import warnings

import numpy as np
from flwr.common import (
    Context,
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from utils.model import *
from utils.data import *

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)
logging.basicConfig(level=logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("flwr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def create_server_app(config_dict, res_dict):

    def get_evaluate_fn(path_data):
        """Return an evaluation function for server-side evaluation."""

        size_win = config_dict['train_param']["win_size"] 
        stride_win = config_dict['train_param']["win_stride"] 
        size_batch = config_dict['train_param']["batch_size"]

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself
        test, label = [], []
        for i in range(config_dict['train_param']['num_edge_nodes']):
            server_train, server_test, test_label = load_data(i+1, path_data)
            if 'QAPPD' in path_data:
                server_train, server_test, test_label = down_samp(server_train, server_test, test_label, factor=2)
            _, server_test = scale_data(server_train, server_test, 'minmax')

            test_win = create_win_periods(
                server_test, size_win, stride_win
            )
            label_win = create_win_periods(
                test_label.reshape(test_label.shape[0], 1), size_win, stride_win
            )

            test.append(test_win)
            label.append(label_win) # .reshape(-1, win_dim, 1)

        test = np.concatenate(test, axis=0)
        label = np.concatenate(label, axis=0)
        flat_ = True if 'USAD' in config['model_name'] else False
        test = batch_loader(test, size_batch, flatten=flat_)
        label = batch_loader(label, size_batch, flatten=flat_)
        test = CombinedLoader([test, label], mode="max_size_cycle")
        network = gen_net()

        # The `evaluate` function will be called after every round
        def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) \
                -> Optional[Tuple[float, Dict[str, Scalar]]]:
            set_parameters(parameters, network)  # Update model with the latest parameters
            trainer = init_trainer(1, verbose=0, logger=False)
            test_loss = trainer.test(network, test, verbose=False)
            test_loss = test_loss[0]["test-loss"]
            vus_roc = network.results['VUS_ROC']
            vus_pr = network.results['VUS_PR']
            auc_roc  = network.results['AUC_ROC']
            auc_pr = network.results['AUC_PR']
            best_f1_p = network.results['Best-F1-Point-Wise']
            best_f1_c = network.results['Best-F1-Composite']

            res_dict.setdefault('test-loss', []).append(test_loss)
            res_dict.setdefault('VUS_ROC', []).append(vus_roc)
            res_dict.setdefault('VUS_PR', []).append(vus_pr)
            res_dict.setdefault('AUC_ROC', []).append(auc_roc)
            res_dict.setdefault('AUC_PR', []).append(auc_pr)
            res_dict.setdefault('Best-F1-Point-Wise', []).append(best_f1_p)
            res_dict.setdefault('Best-F1-Composite', []).append(best_f1_c)

            return float(0), {"f1-score": best_f1_p}  # float(np.mean(res))

        return evaluate

    def server_fn(context: Context) -> ServerAppComponents:
        """Construct components for ServerApp."""

        data_path = config_dict['data_dir']
        client_num = config_dict['train_param']['total_nodes']
        ndarrays = get_parameters(
            gen_net()
        )
        global_model_init = ndarrays_to_parameters(ndarrays)

        def aggregate_fit_metrics(results):
            """Aggregate evaluation results obtained from multiple clients."""
            num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
            weighted_losses = [num_examples * loss['loss'] for num_examples, loss in results]
            total_loss = sum(weighted_losses) / num_total_evaluation_examples
            res_dict.setdefault("loss", []).append(total_loss)
            return {"wighted_avg_train_loss":total_loss}

        strategy = FedAvg(  
            fraction_fit=client_num,  # this is a fraction of scheduled nodes. e.g. 0.15
            fraction_evaluate=0.0,
            initial_parameters=global_model_init,
            evaluate_fn=get_evaluate_fn(data_path),  # server-side evaluation
            # weighted_by_key="loss",
            fit_metrics_aggregation_fn=aggregate_fit_metrics

        )
        num_rounds = config_dict['train_param']["num_server_rounds"]
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return ServerApp(server_fn=server_fn)
# server = ServerApp(server_fn=server_fn)
