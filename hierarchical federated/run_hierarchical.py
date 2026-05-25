import os

import pandas as pd

from hierarchical_federated.server_controller import ServerController, ray
from hierarchical_federated.model_trainer import ModelTrainer
from utils.model import config, gen_net


def run_hierarchical():
    if ray.is_initialized():
        ray.shutdown()  
    ray.init(log_to_driver=False)

    model = gen_net()

    server_simulator = ServerController(model, ModelTrainer, config)
    result = server_simulator.run()

    csv_filename = os.path.join(config['out_dir'], f'{config['model_name']}_{config['dataset_name']}_hfl/')
    os.makedirs(csv_filename, exist_ok=True)
    df = pd.DataFrame(result)
    df.to_csv(f'{csv_filename}metrics.csv', index=False)
