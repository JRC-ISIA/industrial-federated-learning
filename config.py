import yaml


dataset = 'QAPPD'
model = 'DeepAnT'
paradigm = 'H-FL'

with open(f"configs/{model.lower()}_{paradigm.lower()}_{dataset.lower()}.yaml") as f:
    config = yaml.safe_load(f)

model_params = config["model_params"]
train_params = config["train_params"]

config = {
    "paradigm": paradigm,
    "model_param": model_params,
    "train_param": train_params,
    "data_dir": f'datasets/{dataset}/',
    "out_dir": 'results/',
    'dataset_name': dataset,
    'model_name': model,
    'centralized_val': True
}
