# DualNetGNN_large

Implementation of Dual-Net GNN for large graph datasets from LINKX paper. For details, please refer to our [short paper](https://dl.acm.org/doi/10.1145/3511808.3557543).

Implementation for small graph datasets is available at [this](https://github.com/sunilkmaurya/DualNetGNN) repository.

Experiments were conducted with following setup:

Pytorch: 1.12.0

Python: 3.10.4

Cuda: 11.3.1 trained on A100 GPU (40 GB)

**Summary of Results**

Original data-splits (five) from LINKX paper's repository are used.

| **Dateset**   | **Mean Scores** |
| :------------ | :-------------: |
| Penn94        | 86\.09          |
| pokec         | 81\.55          |
| arXiv-year    | 62\.65          |
| snap-patents  | 70\.22          |
| genius        | 91\.23          |
| twitch-gamers | 66\.36          |

Test score values are reported on lowest validation loss.
Scores are accuracy(%) values for all datasets except genius where rocauc is calculated.

**Preprocessing data**

Run `python preprocessing_code.py` to create the training data for the model. 

Downloaded data is stored in `data` folder and preprocessed files are stored in `processed_data` folder.


**Training**

Execute `run_experiments_linkx.sh` to train the model on all datasets.

Datasets and parts of preprocessing code were taken from [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) repository. We thank the authors of the paper for sharing their code.