from collections import OrderedDict

import torch as t
import lightning as L

from config import config
from models.deepant import DeepAnT
from models.lstmae import LSTMAE
from models.mtadgat import MTADGAT
from models.tranad import TranAD
from models.usad import USAD


def set_parameters(parameters, model):
    """ sets parameters to a model """

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: t.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

    return model


def get_parameters(model):
    """ gets parameters of a model """

    parameters = [
        val.cpu().numpy() for _, val in model.state_dict().items()
    ]

    return parameters


def init_trainer(epochs: int, verbose: int, logger: object):
    """ initializes the Lightning's trainer object with the given number of epochs, verbose and debug flags """

    mode = False if verbose == 0 else True

    return L.Trainer(
        devices='auto', accelerator='auto', log_every_n_steps=1,
        max_epochs=epochs, deterministic=True,
        enable_model_summary=mode, enable_progress_bar=mode,
        logger=logger,
        num_sanity_val_steps=0,
        # logger=False,
        enable_checkpointing=False,
    )

def gen_net():
    """ creates an instance of a model """

    model = config['model_name']
    model_param = config['model_param']

    return eval(model)(**model_param)
