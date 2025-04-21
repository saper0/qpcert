This codebase is used to generate the results for 

1. [Exact Certification of (Graph) Neural Networks Against Label Poisoning](https://arxiv.org/abs/2412.00537) published at ICLR 2025 (Spotlight). 

2. [Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks](https://arxiv.org/abs/2407.10867). A preliminary version appeared at the **AdvML-Frontiers @ NeurIPS 2024** workshop and the full paper can be found on [arXiv](https://arxiv.org/abs/2407.10867).

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

Experiments were run using the [seml](https://github.com/TUM-DAML/seml/tree/master) framework and a corresponding MongoDB installation (see seml-link). However, they can be run independently of seml and a MongoDB installation using the corresponding jupyter notebooks provided in the 'notebooks' directory.  

We define the following experiments:

### Label Certification Experiments (LabelCert)

#### Certify Label Poisoning (Binary)

For fast certification, we allow for parallel processing by certifying each node separately. See Jupyter Notebook: `labelcert_binaryclass_onenode.ipynb`

`seml` details:
* Experiment source file: `exp_certify_label_binaryclass_onenode.py`  
* Experiment configurations: `config/label_certification_binary_onenode/`

To certify all test nodes sequentially without parallization (potentially slower), see the jupyter notebook `labelcert_label_binaryclass.ipynb` and the following `seml` files:

* Experiment source file: `exp_certify_label_binaryclass.py`  
* Experiment configurations: `config/label_certification_binary/`

#### Certify Label Poisoning (Multi-Class)

Jupyter Notebook: `labelcert_multiclass_onenode.ipynb`

`seml` details:
* Experiment source file: `exp_certify_multiclass_onenode.py`  
* Experiment configurations: `config/label_certification_multiclass_onenode/`

For better scalability, we provide an inexact implementation of the certificate as discussed in Appendix A. We refer to the jupyter notebook `labelcert_label_multiclass_onenode_inexact.ipynb` and the `seml` files

* Experiment source file: `exp_certify_multiclass_onenode_inexact.py`  
* Experiment configurations: `config/label_certification_multiclass_onenode_inexact/`

#### Certify Label Poisoning (Collective)

Jupyter Notebook: `test_collective.ipynb`

`seml` details:
* Experiment source file: `exp_certify_collective_label.py`  
* Experiment configurations: `config/label_certification_collective/`

### Data Poisoning and Backdoor Attacks Experiments (QPCert)

#### Certify Poisoning (Binary)

Jupyter Notebook: `test_certify.ipynb`

`seml` details:
* Experiment source file: `exp_certify.py`  
* Experiment configurations: `config/certification/`

#### Certify Poisoning (Multi-Class)

Jupyter Notebook: `test_certify_multiclass.ipynb`

`seml` details:
* Experiment source file: `exp_certify_multiclass.py`  
* Experiment configurations: `config/certification/`

#### Certify Backdooring (Cora-MLb, WikiCSb)

Jupyter Notebook: `exp_certify_backdoor.ipynb`

`seml` details:
* Experiment source file: `exp_certify_backdoor.py`  
* Experiment configurations: `config/certification/`

#### Certify Backdooring (CSBM)

Jupyter Notebook: `test_certify_backdoor_csbm.ipynb`

`seml` details:
* Experiment source file: `exp_certify_backdoor_csbm.py`  
* Experiment configurations: `config/certification/`

#### Adversarial Attack (Tightness Evaluation)

Jupyter Notebook: `test_attack.ipynb`

`seml` details:
* Experiment source file: `exp_attack.py`  
* Experiment configurations: `config/attack/`

### Hyperparameter Search (using Cross Validation)

Jupyter Notebook: `test_hyperpar.ipynb`

`seml` details:
* Experiment source file: `exp_hyperparam.py`  
* Experiment configurations: `config/hyperparams/`

## Cite

You can cite our papers as follows:

```
@inproceedings{
  sabanayagam2025exact,
  title={Exact Certification of (Graph) Neural Networks Against Label Poisoning},
  author={Mahalakshmi Sabanayagam and Lukas Gosch and Stephan G{\"u}nnemann and Debarghya Ghoshdastidar},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=d9aWa875kj}
}

@article{gosch2024provable,
  title={Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks},
  author={Gosch, Lukas and Sabanayagam, Mahalakshmi and Ghoshdastidar, Debarghya and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2407.10867},
  year={2024}
}
```

To cite the **AdvML-Frontiers @ NeurIPS24** workshop paper, please use

```
@article{gosch2024provable,
  title={Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks},
  author={Gosch, Lukas and Sabanayagam, Mahalakshmi and Ghoshdastidar, Debarghya and G{\"u}nnemann, Stephan},
  journal={The 3rd Workshop on New Frontiers in Adversarial Machine Learning, NeurIPS},
  year={2024}
}
```

## Contact

For questions and feedback, feel free to contact

Lukas Gosch, lukas (dot) gosch (at) tum (dot) de  
Mahalakshmi Sabanayagam, m (dot) sabanayagam (at) tum (dot) de


## Other Notes

This codebase contains code fragments from

* [Revisiting Robustness in Graph Machine Learning](https://github.com/saper0/revisiting_robustness/)
* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)

We thank the authors for making their code publicly available.