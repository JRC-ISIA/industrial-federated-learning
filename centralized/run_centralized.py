import logging
import warnings

import numpy as np
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from utils.data import *
from utils.model import *


logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)  # turn lightning's device logging off
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)


def run_centralized():
    win_dim = config['train_param']['win_size']
    win_stride = config['train_param']['win_stride']
    batch_size = config['train_param']['batch_size']
    dataset = config['dataset_name']
    flat = True if 'USAD' in config['model_name'] else False

    # load and preprocess the dataset
    train = []
    test = []
    label = []

    for d in range(1, config['train_param']['total_nodes']+1):

        server_train, server_test, test_label = load_data(d, f'datasets/{dataset.upper()}/')
        if dataset == 'QAPPD':
            server_train, server_test, test_label = down_samp(server_train, server_test, test_label, factor=2)
        server_train, server_test = scale_data(server_train, server_test, 'minmax')

        train_win = create_win_periods(
            server_train, win_dim, win_stride
        )
        test_win = create_win_periods(
            server_test, win_dim, win_stride
        )
        label_win = create_win_periods(
            test_label.reshape(test_label.shape[0], 1), win_dim, win_stride
        )

        train.append(train_win)
        test.append(test_win)
        label.append(label_win)

    train = np.concatenate(train, axis=0)
    test = np.concatenate(test, axis=0)
    label = np.concatenate(label, axis=0)

    # feature_dim = train.shape[-1]

    train_loader = batch_loader(train, batch_size, flatten=flat)
    test_loader = batch_loader(test, batch_size, flatten=flat)
    label_loader = batch_loader(label, batch_size, flatten=flat)
    test_loader = CombinedLoader([test_loader, label_loader], mode="max_size_cycle")

    # initialize model
    network = gen_net()

    # initialize trainer
    logger = CSVLogger(
        save_dir=config['out_dir'],
        name=f"{config['model_name']}_{config['dataset_name']}_cl"
    )
    trainer = init_trainer(
        epochs=config['train_param']['epochs'], verbose=0, logger=logger
    )

    # train and evaluate the model
    trainer.fit(network, train_loader, test_loader)
