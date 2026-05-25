import os
import logging

import pandas as pd
from flwr.simulation import run_simulation

from federated.server_app import create_server_app
from federated.client_app import create_client_app
from config import config


def run_federated():
    result = {}  # leverage Python’s pass-by-reference behavior for mutable objects and update in server. (shared state) be careful of this just like global vars in cuncurrent runs (shouldn't be modified somewhere in the middle).
    server = create_server_app(config, result)
    client = create_client_app(config)
    # Run Simulation
    try:
        
        cpus_per_client = max(
            1, os.cpu_count() // config['train_param']['num_edge_nodes']
            )

        run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=config['train_param']['total_nodes'],
            backend_config={"client_resources": {"num_cpus": cpus_per_client, "num_gpus": 1.0}, "init_args": {
            # backend_config={"client_resources": {"num_cpus": cpus_per_client}, "init_args": {
                "log_to_driver": False,
                "logging_level": "ERROR"
            }},
            verbose_logging = False
        )
    except Exception as e:
        logging.error(f"Simulation failed: {e}")

    for key in ["VUS_ROC", "VUS_PR", "AUC_ROC", 'AUC_PR', 'Best-F1-Point-Wise', 'Best-F1-Composite', "test-loss"]: # to pop out the round 0 values
        result[key] = result[key][1:]

    csv_filename = os.path.join(config['out_dir'], f'{config['model_name']}_{config['dataset_name']}_fl/')
    os.makedirs(csv_filename, exist_ok=True)
    df = pd.DataFrame(result)
    df.to_csv(f'{csv_filename}metrics.csv', index=False)
