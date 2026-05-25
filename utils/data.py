import os
import pickle

import numpy as np
import torch as t
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from config import config


def load_data(idx, data_dir):

    """ reads and loads dataset into numpy array """
    f = open(os.path.join(data_dir, f'train_{idx}.pkl'), "rb")
    train_data = pickle.load(f)
    f.close()
    train_data = np.asarray(train_data, dtype=np.float32)

    f = open(os.path.join(data_dir, f'test_{idx}.pkl'), "rb")
    test_data = pickle.load(f)
    f.close()
    test_data = np.asarray(test_data, dtype=np.float32)

    f = open(os.path.join(data_dir, f'test_label_{idx}.pkl'), "rb")
    test_label = pickle.load(f)
    f.close()
    test_label = np.asarray(test_label, dtype=np.float32)
   
    return train_data, test_data, test_label


def down_samp(data_train, data_test, label_test, factor=2):

    """ downsamples the dataset with the factor a given factor """
    data_train = data_train[::factor]
    data_test = data_test[::factor]
    label_test = label_test[::factor]
    
    return data_train, data_test, label_test


def scale_data(data_train, data_test, scaler_type='minmax'):
    """ standardizes or normalized dataset z-score/minmax scaling """

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise Exception(f'{scaler_type} scaler is not implemented')
    train_scaled = scaler.fit_transform(data_train)
    test_scaled = scaler.transform(data_test)

    return train_scaled, test_scaled



def get_datasets_stats(config_, sample_idx):
    """ loads and calculates training samples"""

    flat = True if 'USAD' in config['model_name'] else False
    dataset_path = config_['data_dir']
    batch_size = config_['train_param']['batch_size']
    win_size = config_['train_param']['win_size']
    win_stride = config_['train_param']['win_stride']
    train_set, val_set, label_set = load_data(sample_idx, dataset_path)
    train_set, val_set = scale_data(train_set, val_set, 'minmax')
    train_set = create_win_periods(train_set, win_size, win_stride)
    num_train_samples = len(train_set)
    test_set = create_win_periods(val_set, win_size, win_stride)
    train_set = batch_loader(train_set, batch_size, flatten=flat)
    test_set = CombinedLoader([test_set, label_set], mode="max_size_cycle")

    return train_set, test_set, num_train_samples


def batch_loader(data_, batch_size_, flatten):
    """ creates a data loader for batch processing """

    if not isinstance(data_, t.Tensor):
        data_ = t.from_numpy(data_.copy())
    if flatten:
        data_ = data_.contiguous().view(data_.shape[0], -1)
    data_ = data.DataLoader(
        data_, batch_size=batch_size_, shuffle=False,
        # num_workers=int(os.cpu_count() / 2), persistent_workers=True, pin_memory=True)
        num_workers=2)

    return data_


def create_win_periods(data_, win_size_, win_stride_):
    """ returns the rolling windows of the given flattened data """

    windows = sliding_window_view(
        data_, (win_size_, data_.shape[1])
    )

    return windows.squeeze()[::win_stride_, :]


