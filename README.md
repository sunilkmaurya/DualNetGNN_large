# DualNetGNN_large

Implementation of Dual-Net GNN for large graph datasets from LINKX paper.

Experiments were conducted with following setup:

Pytorch: 1.12.0

Python: 3.10.4

Cuda: 11.3.1

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


Scores are accuracy(%) values for all datasets except genius where rocauc is calculated.

**Preprocessing data**

Run `python preprocessing_code.py` to create the training data for the model.


**Training**

Execute `run_experiments_linkx.sh` to train the model on all datasets.