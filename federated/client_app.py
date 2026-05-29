import logging
import warnings

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

from utils.data import *
from utils.model import *

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
    logging.FATAL)  # turns off Lightning's device logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("flwr").setLevel(logging.ERROR)


class FlowerClient(NumPyClient):

    def __init__(self, train_set, epoch):
        self.network = gen_net()
        self.train_set = train_set
        self.epoch = epoch
        # self.server_round = round

    def fit(self, parameters, config):
        set_parameters(self.network, parameters)
        trainer = init_trainer(self.epoch, verbose=0, logger=False)
        trainer.fit(self.network, self.train_set)
        fit_metrics = trainer.callback_metrics
        loss = float(fit_metrics["loss"])
        return get_parameters(self.network), len(self.train_set), {"loss": float(loss)}


def create_client_app(config_dict):

    def client_fn(context: Context) -> Client:
        idx = int(context.node_config["partition-id"])  # no partitioning, only used as clients IDs
        data_path = config_dict["data_dir"]
        client_train, client_test, client_label = load_data(idx + 1, data_path)
        if 'QAPPD' in data_path:
            client_train, client_test, _ = down_samp(client_train, client_test, client_label, factor=2)
        win_size = config_dict["train_param"]["win_size"]
        win_stride = config_dict["train_param"]["win_stride"]
        batch_size = config_dict["train_param"]["batch_size"]
        epochs = config_dict["train_param"]["local_epochs"]
        flat_ = True if 'USAD' in config['model_name'] else False
        client_train, _ = scale_data(client_train, client_test, 'minmax')
        client_train = create_win_periods(client_train, win_size, win_stride)
        client_train = batch_loader(client_train, batch_size, flatten=flat_)
        return FlowerClient(client_train, epochs).to_client()

    return ClientApp(client_fn=client_fn)
# client = ClientApp(client_fn=client_fn)
