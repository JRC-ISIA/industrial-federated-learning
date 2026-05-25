from hierarchical_federated.client_trainer import ClientTrainer
from utils.data import get_datasets_stats
from utils.model import init_trainer


class ModelTrainer(ClientTrainer):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.train_set = None
        self.test_set = None
        self.num_train_samples = None

    def get_data_and_stats(self):
        self.train_set, self.test_set, self.num_train_samples = get_datasets_stats(
            self.config, self.sample_id
        )

    def train(self, device):
        # logging.info('Client_ID {} starting to train.'.format(self.client_id))
        self.get_data_and_stats()
        trainer = init_trainer(self.config['train_param']['local_epochs'], verbose=False, logger=False)
        trainer.fit(self.model, self.train_set)
        total_train_loss = trainer.callback_metrics.get("loss")
        return self.num_train_samples, total_train_loss

    def test(self, device):
        pass
    
