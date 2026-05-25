from abc import ABC, abstractmethod
from collections import OrderedDict

from hierarchical_federated.server_controller import t


class ClientTrainer(ABC):
    """
    Abstract base class for client trainers of federated learning simulation
    """

    def __init__(self, model):
        self.local_model = model
        self.client_id = None
        self.sample_id = None
        self.device = None
        self.config = None

    def set_client_id(self, client_id):
        """ Sets client index attribute """
        self.client_id = client_id

    def set_sample_id(self, sample_id):
        """ Sets sample or subsample (dataset or partition) index attribute """
        self.sample_id = sample_id

    def set_config(self, config):
        self.config = config

    # @abstractmethod
    def set_parameters(self, parameters):
        params_dict = zip(self.local_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: t.tensor(v) for k, v in params_dict})
        self.local_model.load_state_dict(state_dict, strict=True)

    # @abstractmethod
    def get_parameters(self):
        parameters = [
            val.cpu().numpy() for _, val in self.local_model.state_dict().items()
        ]
        return parameters

    @abstractmethod
    def train(self, device):
        pass

    @abstractmethod
    def test(self, device):
        pass
    
