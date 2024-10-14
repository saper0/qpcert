# Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoors

Codebase to generated the results for the ICML 2025 submission "Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoors".

## Installation

The codebase has been developed using Python 3.11 and requires the following packages and has been tested with the listed versions:

* pytorch 2.0.1
* pyg (pytorch geometric) 2.3.1
* torch-scatter 2.1.1.+pt20cu118
* torch-sparse 0.6.17+pt20cu118
* qpth 0.0.16
* gurobi 11.0.1
* jaxtyping 0.2.22
* typeguard 4.1.4
* networkx 3.1
* numpy 1.25.2
* seml 0.4.0
* scikit-learn 1.2.2
* cvxopt 1.3.2
* jupyterlab 4.0.6
* pandas 2.1.3

## Experiment Files

Experiments were run using the [seml](https://github.com/TUM-DAML/seml/tree/master) framework and a corresponding MongoDB installation (see seml-link). However, they can be run independently of seml and a MongoDB installation using the corresponding jupyter notebooks provided in the root directory and prefixed with `test_`.  

We define the following experiments:

### Certify Poisoning (Binary)

Jupyter Notebook: `test_certify.ipynb`

`seml` details:
* Experiment source file: `exp_certify.py`  
* Experiment configurations: `config/certification/`

### Certify Poisoning (Multi-Class)

Jupyter Notebook: `test_certify_multiclass.ipynb`

`seml` details:
* Experiment source file: `exp_certify_multiclass.py`  
* Experiment configurations: `config/certification/`

### Certify Backdooring (Cora-MLb, WikiCSb)

Jupyter Notebook: `exp_certify_backdoor.ipynb`

`seml` details:
* Experiment source file: `exp_certify_backdoor.py`  
* Experiment configurations: `config/certification/`

### Certify Backdooring (CSBM)

Jupyter Notebook: `test_certify_backdoor_csbm.ipynb`

`seml` details:
* Experiment source file: `exp_certify_backdoor_csbm.py`  
* Experiment configurations: `config/certification/`

### Adversarial Attack (Tightness Evaluation)

Jupyter Notebook: `test_attack.ipynb`

`seml` details:
* Experiment source file: `exp_attack.py`  
* Experiment configurations: `config/attack/`

### Hyperparameter Search (using Cross Validation)

Jupyter Notebook: `test_hyperpar.ipynb`

`seml` details:
* Experiment source file: `exp_hyperparam.py`  
* Experiment configurations: `config/hyperparams/`

## Other Notes

This codebase contains code fragments from

* [Revisiting Robustness in Graph Machine Learning](https://github.com/saper0/revisiting_robustness/)
* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)

We thank the authors for making their code publicly available.