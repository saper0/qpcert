# Exact Certification of (Graph) Neural Networks Against Label Poisoning

This repository provide the code to reproduce the results for the ICLR 2025 submission "Exact Certification of (Graph) Neural Networks Against Label Poisoning".

## Installation

The codebase is built using the NTKs from the implementation of [Provable robustness of (graph) neural networks against data poisoning and backdoor attacks](https://arxiv.org/abs/2407.10867). We thank the authors for sharing the repository with us. The code is developed using Python 3.11 and requires the following packages and has been tested with the listed versions:

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

### Certify Label Poisoning (Binary)

To speed up the certification, we do parallel processing by certifying each node separately. Check Jupyter Notebook: `test_certify_label_binaryclass_onenode.ipynb`

`seml` details:
* Experiment source file: `exp_certify_label_binaryclass_onenode.py`  
* Experiment configurations: `config/label_certification_binary_onenode/`

Check Jupyter Notebook: `test_certify_label_binaryclass.ipynb` for implementation without parallel processing.

`seml` details:
* Experiment source file: `exp_certify_label_binaryclass.py`  
* Experiment configurations: `config/label_certification_binary/`


### Certify Label Poisoning (Multi-Class)

Jupyter Notebook: `test_certify_label_multiclass_onenode.ipynb`

`seml` details:
* Experiment source file: `exp_certify_multiclass_onenode.py`  
* Experiment configurations: `config/label_certification_multiclass_onenode/`

To improve the scalability, we provide an inexact implementation of the certificate as discussed in Appendix A. See Jupyter Notebook: `test_certify_label_multiclass_onenode_inexact.ipynb`

`seml` details:
* Experiment source file: `exp_certify_multiclass_onenode_inexact.py`  
* Experiment configurations: `config/label_certification_multiclass_onenode_inexact/`

### Certify Label Poisoning (Collective)

Jupyter Notebook: `test_collective.ipynb`

`seml` details:
* Experiment source file: `exp_certify_collective_label.py`  
* Experiment configurations: `config/label_certification_collective/`

### Hyperparameter Search (using Cross Validation)

Jupyter Notebook: `test_hyperparam.ipynb`

`seml` details:
* Experiment source file: `exp_hyperparam.py`  
* Experiment configurations: `config/hyperparams/`

## Code information

Our code is an adaptation and extension of the codebase from

* [Provable robustness of (graph) neural networks against data poisoning and backdoor attacks](https://arxiv.org/abs/2407.10867) 

Furthermore, it contains code fragments from

* [Revisiting Robustness in Graph Machine Learning](https://github.com/saper0/revisiting_robustness/)
* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)