# Federated Learning for Multivariate Time Series Anomaly Detection in Industrial Automation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20287835.svg)](https://doi.org/10.5281/zenodo.20287835)

This repository contains the code and dataset for our paper "Federated Learning for Multivariate Time Series Anomaly Detection in Industrial Automation". The paper presents a novel approach to anomaly detection in industrial automation using federated learning on multivariate time series data. 
The dataset used in our experiments, the Quanser Aero 2 Pick-and-Place Dataset (QAPPD), which is available for download on Zenodo, contains multivariate time series data collected from a Quanser Aero 2 system performing pick-and-place tasks under various conditions, including normal operation and different types of anomalies.

The code in this repository includes the implementation of our federated learning approach for anomaly detection, as well as scripts for data preprocessing, model training, and evaluation. We also provide instructions on how to reproduce our experiments and results.


## Abstract
Federated learning has opened new possibilities for multivariate time series anomaly detection, but 
reliable benchmarking remains difficult because existing datasets often lack scale, accurate labels, 
or clean experimental conditions. To address this, we introduce a new cyclic dataset for industrial 
automation and evaluate selected anomaly detection methods on both the proposed dataset and public 
benchmarks across centralized, federated, and hierarchical federated settings. 
Our results show that model performance depends strongly on both the learning paradigm and the dataset 
characteristics, highlighting the need for domain-aware evaluation in industrial anomaly detection.

## Repository Structure
```

├── centralized
|  └── run_centralized.py
|
├── configs
|  ├── deepant_cl_asd.yaml
|  ├── deepant_cl_qappd.yaml
|  ⋮     ⋮
|
├── datasets
|  └── placeholder
|
├── federated
|  ├── client_app.py
|  ├── server_app.py
|  └── run_federated.py
|
├── hierarchical_federated
|  ├── client_controller.py
|  ├── client_trainer.py
|  ├── model_trainer.py
|  ├── run_hierarchical.py
|  ├── server_aggregator.py
|  └── server_controller.py
|
├── models
|  ├── deepant.py
|  ├── lstmae.py
|  ├── mtadgat.py
|  ├── tranad.py
|  └── usad.py
|
├── utils
|  ├── __init__.py
|  ├── data.py
|  ├── eval.py
|  └── model.py
|
├── CITATION.cff
├── config.py
├── LICENSE
├── main.py
└── README.md
└── requirements.txt
```

## Installation
From the root directory, install the required dependencies using:

```
pip install -r requirements.txt
```

The requirements.txt file includes the following packages:

```
flwr==1.30.0
lightning==2.6.4
matplotlib==3.10.9
numpy==2.4.6
pandas==3.0.3
PyYAML==6.0.3
scikit-learn==1.8.0
scipy==1.17.1
torch-geometric==2.7.0
torchmetrics==1.9.0
```

If dependency conflicts arise, execute the following commands sequentially:

``` 
pip uninstall -y numpy pandas scipy scikit-learn matplotlib lightning torchmetrics
pip install numpy pandas scipy scikit-learn matplotlib lightning torchmetrics
```

## Reproducing Paper Results
Before running the experiments, ensure that all dependencies are correctly installed as described above. <br>

Next, configure the experiment by modifying configs.py to specify the desired dataset, model, and learning paradigm from the following options:

``` 
Datasets:
    ASD
    QAPPD

Models:
    DeepAnT
    LSTMAE
    MTADGAT
    TranAD
    USAD

Paradigms:
    CL    (Centralized Learning)
    FL    (Federated Learning)
    H-FL  (Hierarchical Federated Learning)
 ```

Finally, execute the main script:

```
$ python main.py
```

## Data
The Quanser Aero 2 Pick-and-Place Dataset (QAPPD) is available for download on Zenodo at the following link: 
[https://doi.org/10.5281/zenodo.20287835](https://doi.org/10.5281/zenodo.20287835). 
The dataset includes multivariate time series data collected from a Quanser Aero 2 system performing 
pick-and-place tasks under various conditions, including normal operation and different types of 
anomalies.


The Application Server Dataset (ASD) is available in the 
[InterFusion GitHub repository](https://github.com/zhhlee/InterFusion/tree/main).


Please download both datasets and place them in the `datasets` directory, ensuring the following structure:

 ```
datasets/
├── ASD/
│   ├── train_0, train_1, ...
│   ├── test_0, test_1, ...
│   └── test_label_0, test_label_1, ...
└── QAPPD/
    ├── train_0, train_1, ...
    ├── test_0, test_1, ...
    └── test_label_0, test_label_1, ...
```


All files should follow an incremental naming convention (e.g., `train_0`, `train_1`, ..., and similarly for 
`test` and `test_label`).

## Citation
If you find this work useful in your research, please consider citing it:

```bibtex
@inproceedings{nosrati2026,
  author = {Nosrati, Khayyam and Uray, Martin and Messineo, Saverio and Sassnick, Olaf and Huber, Stefan},
  title  = {Federated Learning for Multivariate Time Series Anomaly Detection in Industrial Automation},
  booktitle = {Database and Expert Systems Applications - DEXA 2026 Workshops: AISys and AI4IP},
  volume = {},
  number = {},
  pages  = {},
  year   = {2026},
  month  = Aug,
  note   = {To be published},
  url    = {http://arxiv.org/abs/2605.XXXXX}
}

@dataset{nosrati2026dataset,
  author       = {Nosrati, Khayyam and Uray, Martin and Messineo, Saverio and Sassnick, Olaf and Huber, Stefan},
  title        = {Quanser Aero 2 Pick-and-Place Dataset (QAPPD)},
  month        = May,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.20287835},
  url          = {https://doi.org/10.5281/zenodo.20287835},
}
```

## License

This code of this project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

The financial support by the Austrian Federal Ministry of Economy, Energy
and Tourism, the National Foundation for Research, Technology and
Development and the Christian Doppler Research Association is gratefully
acknowledged.

## Contact

Martin Uray: martin.uray \[at\] fh-salzburg.ac.at  
Open an issue for bug reports or questions.
