# ntk-robust

## Installation

Current implementation requires at least pytorch 2.0.0, I use it with pytorch 2.0.1 and cuda 11.8.

It further requires a fitting pytorch geometric, torch-scatter and torch-sparse installation to the pytorch version. For torch-scatter and torch-sparse, the conda install failed and I had to 
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html (details see github pages for torch-scatter and torch-sparse packages)

## Other notes

Currently, a transductive poisioning scenario, where only the unlabeled nodes are attacked is implemented. TODO: Change to inductive evasion.