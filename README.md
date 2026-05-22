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
 TODO

## Installation
  TODO

## Reproducing Paper Results
  TODO

## Data
The Quanser Aero 2 Pick-and-Place Dataset (QAPPD) is available for download on Zenodo at the following link: 
[https://doi.org/10.5281/zenodo.20287835](https://doi.org/10.5281/zenodo.20287835). 
The dataset includes multivariate time series data collected from a Quanser Aero 2 system performing 
pick-and-place tasks under various conditions, including normal operation and different types of 
anomalies.


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
