from typing import Any
from numpy.typing import NDArray

import numpy as np


class ServerAggregator:
    """
    Class for fog and cloud server aggregators of federated learning simulation
    """
    
    @staticmethod
    def aggregate(
            parameters_list: list[list[NDArray]],
            local_samp_counts: list[int]
    ) -> tuple[list[NDArray], Any]:  # parameters_or_grads_list
        """
        Computes and returns weighted-averaged parameters of the given models (FedAvg algorithm)
        parameters_list: list of model parameters to be aggregated
        local_samp_counts: list of number of samples used for local training or fog aggregation
        """

        averaged_params = [
            np.zeros_like(layer) for layer in parameters_list[0]  # fist parameter space since it's a homogenous setup
        ]
        global_samp_count = np.sum(local_samp_counts)

        for model_param, local_samp_count in zip(parameters_list, local_samp_counts):
            weight_factor = local_samp_count / global_samp_count

            for idx, layer_param in enumerate(model_param):
                averaged_params[idx] += weight_factor * layer_param  # weighted sum

        return averaged_params, global_samp_count
        
