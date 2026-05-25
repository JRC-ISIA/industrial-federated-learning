import logging
from re import T
import warnings

import numpy as np
import lightning as L
import torch as t
import ray

from hierarchical_federated.client_controller import ClientController
from hierarchical_federated.server_aggregator import ServerAggregator
from utils.data import scale_data, load_data, create_win_periods, batch_loader, CombinedLoader
from utils.model import get_parameters, set_parameters

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)  # turn lightning's device logging off
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)
from numpy.exceptions import VisibleDeprecationWarning
# Filter out the specific VisibleDeprecationWarning
warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)


class ServerController:
    """
    Server controller class for federated learning simulation
    """

    def __init__(self, global_model, client_trainer, config):
        self.global_model = global_model
        self.global_param = get_parameters(global_model)
        self.config = config  # for grid search 
        self.report = {}
        self.client_controller = None
        self.client_trainer = client_trainer
        self.global_round = 1  # global communication rounds
        self.fog_shards = []
        self.client_controller = None
        self.device = t.device(  # not recommended if used in lightning but works
            'cuda' if t.cuda.is_available() else 'cpu'
        )

    def init(self):
        """ Initiates clients """
        assert self.config['train_param']['num_fog_servers'] >= 2, 'At least 2 fog servers are required'  
        client_count = self.config['train_param']['total_nodes']
        # Generate unique client IDs and shuffle them once; stay fixed together
        idx_client = np.arange(client_count)
        idx_sample = np.random.permutation(np.arange(1, client_count + 1))
        full_table = np.column_stack((idx_sample, idx_client))
        shuffled_table = np.random.permutation(full_table)
        # Split both into disjoint shards
        self.fog_shards = np.array_split(shuffled_table, self.config['train_param']['num_fog_servers'])
        
        available_cpus = ray.cluster_resources().get("CPU", 1)
        available_gpus = ray.cluster_resources().get("GPU", 1)
        
        target_parallelism = self.config['train_param']['num_fog_servers']# * self.config['train_param']['num_edge_nodes']

        cpu_fraction = available_cpus / target_parallelism
        gpu_fraction = available_gpus / target_parallelism

        # Put large objects in shared memory to avoid slow copying
        model_ref = ray.put(self.global_model)
        weights_ref = ray.put(self.global_param)  # this is static throughout the simulation, so only once.

        self.client_controller = []
        for s in range(self.config['train_param']['num_fog_servers']):
            
            shard_data = self.fog_shards[s]            
            # Initial random sample for Round 1 from this shard only
            rand_idx = np.random.choice(len(shard_data), self.config['train_param']['num_edge_nodes'], replace=False)
            scheduled_clients = shard_data[rand_idx, :]
    
            self.client_controller.append(
                ClientController.options(
                    num_gpus=gpu_fraction,
                    num_cpus=cpu_fraction
                ).remote(
                    global_model=model_ref,
                    global_weights=weights_ref,
                    scheduled_clients=scheduled_clients,
                    client_trainer=self.client_trainer,
                    config=self.config,
                    device=self.device
                )
            )

    def run(self):
        """" Runs federated learning simulation """
        # logging.info('Hierarchical Federated Learning Simulation is being initiated with {} fog servers'.format(
        #     CONFIG['num_fog_servers'])
        # )
        # print('Hierarchical Federated Learning Simulation is being initiated with {} fog servers'.format(
        #     self.CONFIG['num_fog_servers'])
        # )

        self.init()

        total_fog_samp_num = None
        for i in range(self.config['train_param']['num_server_rounds']):
            # logging.info('Global round {}:'.format(self.global_round))
            # print('Global round {}:'.format(self.global_round))
            
            # return a list of ObjectRefs (futures)
            train_futures = [ctrl.train.remote() for ctrl in self.client_controller]

            # Wait for ALL fog servers to finish and collect results
            results = ray.get(train_futures)

            # Process the collected results
            fog_updates = []
            fog_num_data_list = []
            fog_loss_updates = []
            for local_weights, num_client_samp_list, local_loss in results:
                local_update, total_fog_samp_num = self._aggregate(local_weights, num_client_samp_list)
                local_avg_loss = self._avg_loss(num_client_samp_list, local_loss)  # orders aligned? todo: create a dict with keys as client ids to avoid mismatch (little comutational inefficiency but neglegible)
                fog_updates.append(local_update)
                fog_num_data_list.append(total_fog_samp_num)
                fog_loss_updates.append(local_avg_loss)

            # Global Aggregation
            cloud_update, _ = self._aggregate(fog_updates, fog_num_data_list)
            total_train_loss = self._avg_loss(fog_num_data_list, fog_loss_updates)
            total_train_loss = total_train_loss.item()
            self.report.setdefault('train-loss', []).append(total_train_loss)
            self.global_param = cloud_update
            
            # Prepare for the next round (if not the last round)
            if i < self.config['train_param']['num_server_rounds'] - 1:
                weights_ref = ray.put(self.global_param)

                # Tell existing actors to refresh for the next round
                # List of remote tasks
                update_tasks = []
                for s in range(self.config['train_param']['num_fog_servers']):
                    # Select clients for this specific fog server
                    shard = self.fog_shards[s]
                    
                    rand_idx_ = np.random.choice(len(shard), self.config['train_param']['num_edge_nodes'], replace=False)
                    new_schedule = shard[rand_idx_, :]
                    
                    # Dispatch the update to the actor
                    update_tasks.append(
                        self.client_controller[s].start_new_round.remote(weights_ref, new_schedule)
                    )
                ray.get(update_tasks)
            
            self.global_round += 1

            # load test data here if centralized evaluation is needed
            if self.config['centralized_val']:
                window_size = self.config['train_param']['win_size']
                window_stride = self.config['train_param']['win_stride']
                batch_size = self.config['train_param']['batch_size']

                test, label = [], []
                for j in range(self.config['train_param']['total_nodes']):
                    server_train, server_test, test_label = load_data(j+1, self.config['data_dir'])
                    _, server_test = scale_data(server_train, server_test, 'minmax')
        
                    test_win = create_win_periods(
                        server_test, window_size, window_stride
                    )
                    label_win = create_win_periods(
                        test_label.reshape(test_label.shape[0], 1), window_size, window_stride
                    )
        
                    test.append(test_win)
                    label.append(label_win)
        
                test = np.concatenate(test, axis=0)
                label = np.concatenate(label, axis=0)
                
                flat_ = True if 'USAD' in self.config['model_name'] else False
                test = batch_loader(test, batch_size, flatten=flat_)
                label = batch_loader(label, batch_size, flatten=flat_)
                test = CombinedLoader([test, label], mode="max_size_cycle")
            
            self.test(test)
     
        for ctrl in self.client_controller:
            ray.kill(ctrl) 
                
        return self.report
        
    def _avg_loss(self, num_samp, losses):
        total_samples = sum(num_samp)
        weighted_loss = sum((n / total_samples) * loss for n, loss in zip(num_samp, losses))
        return weighted_loss

    def _aggregate(self, weights_list, num_client_samp_list):
        """ Aggregates clients model parameters"""
        updated_weights, global_samp_num = ServerAggregator.aggregate(weights_list, num_client_samp_list)  #self.global_param
        return updated_weights, global_samp_num

    def test(self, test_loader):
        """ Server-side (global) test """
        # logging.info('Running server-side test:')
        # print('\t Running server-side test')
        self.global_model = set_parameters(self.global_param, self.global_model)
        trainer = L.Trainer(devices=1, accelerator='auto', num_nodes=1, logger=False, enable_progress_bar=False)
        test_loss = trainer.test(self.global_model, test_loader, verbose=False)
        test_loss = test_loss[0]["test-loss"]
        vus_roc = self.global_model.results['VUS_ROC']
        vus_pr = self.global_model.results['VUS_PR']
        auc_roc  = self.global_model.results['AUC_ROC']
        auc_pr = self.global_model.results['AUC_PR']
        best_f1_p = self.global_model.results['Best-F1-Point-Wise']
        best_f1_c = self.global_model.results['Best-F1-Composite']

        self.report.setdefault('VUS_ROC', []).append(vus_roc)
        self.report.setdefault('VUS_PR', []).append(vus_pr)
        self.report.setdefault('AUC_ROC', []).append(auc_roc)
        self.report.setdefault('AUC_PR', []).append(auc_pr)
        self.report.setdefault('Best-F1-Point-Wise', []).append(best_f1_p)
        self.report.setdefault('Best-F1-Composite', []).append(best_f1_c)
        # print(
        #     'Server-side evaluation results: Test loss: {}\tf1 score: {}'.format(test_loss, f1_score)
        # )
        # logging.info(
        #     'Server-side evaluation results: Test loss: {}\tf1 score: {}'.format(
        #         self.global_model.results[0]['loss'], self.global_model.results[0]['f1']
        #    )
        # )
        
