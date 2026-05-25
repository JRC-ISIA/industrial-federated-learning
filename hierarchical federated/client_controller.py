import warnings

from hierarchical_federated.server_controller import ray


@ray.remote
class ClientController:
    """
    Class for federated learning server controller
    """

    def __init__(
            self, global_model, global_weights, scheduled_clients, client_trainer, config, device
    ):
            
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*num_workers.*")
        self.weights = global_weights
        self.local_weights_list = []
        self.sample_idx = iter(scheduled_clients[:, 0])
        self.client_idx = iter(scheduled_clients[:, 1])
        self.client_ids = scheduled_clients[:, 1]
        self.sample_ids = scheduled_clients[:, 0]
        # Each Actor lives in its own process with its own trainer instance.
        # Ray handles parallel execution and GPU device isolation automatically.
        self.client_trainer = client_trainer(global_model)  # fixme: if not ray: this should be instantiated multiple times for multiple threads; todo: device scheduling
        self.config = config
        self.device = device

    def get_client_stats(self):
        """Returns the IDs currently scheduled for this actor"""
        return self.client_ids, self.sample_ids
        
    def start_new_round(self, weights, scheduled_clients):
        """ This is what the server calls to refresh the actor """
        self.weights = weights
        self.local_weights_list = [] # Reset for new round
        self.sample_idx = iter(scheduled_clients[:, 0])
        self.client_idx = iter(scheduled_clients[:, 1])
        self.client_ids = scheduled_clients[:, 1]
        self.sample_ids = scheduled_clients[:, 0]

    def update_client(self):
        """ Local data info updates making it possible for standard or continual federated learning """
        self.client_trainer.set_parameters(self.weights)
        self.client_trainer.set_sample_id(next(self.sample_idx))
        self.client_trainer.set_client_id(next(self.client_idx))
        self.client_trainer.set_config(self.config)

    def get_dataset_stats(self):
        return

    def train(self):
        """ Train client models """
        num_train_samp_list = []
        train_loss_total = []
        for _ in range(len(self.client_ids)):
            self.update_client()
            num_train_samp, train_loss = self.client_trainer.train(self.config)
            local_weights = self.client_trainer.get_parameters()
            self.local_weights_list.append(local_weights)
            num_train_samp_list.append(num_train_samp)
            train_loss_total.append(train_loss)
        return self.local_weights_list, num_train_samp_list, train_loss_total

    def test(self):
        """ Client-side test """
        metrics = self.client_trainer.test(self.config)
        return metrics
