from collections.abc import Iterable
import copy
import math
from pymongo import MongoClient
from typing import Any, Dict, Iterator, List, Tuple, Union

from cycler import cycler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import rankdata
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
import networkx as nx
import sys
sys.path.append('../../')
# sys.path.insert(1, 'ntk-robust/src')
from src.data import get_karate_club

CERTIFICATE_FIGURE_DIR = Path('./figures')

URI = "mongodb://sabanaya:bAvQwbOp@fs.kdd.in.tum.de:27017/sabanaya?authMechanism=SCRAM-SHA-1"


def assert_equal_dicts(d1: Dict[str, Any], d2: Dict[str, Any]):
    """Checks if every key of d1 is in d2 and they store same value."""
    for key in d1:
        assert key in d2
        assert d1[key] == d2[key]


def append_dict(source: Dict[str, Any], target: Dict[str, Any], 
                exclude_keys: List[str] = []):
    """Appends each element in source-dict to same key-element in target. 
    
    Assumes source-dict has as elements a single number, list, or a dict. 
    Possibility to ignore keys given in exclude_keys.
    """
    for key, item in source.items():
        if key not in exclude_keys:
            if isinstance(item, dict):
                if key not in target:
                    target[key] = {}
                append_dict(item, target[key])
            elif isinstance(item, list):
                if key not in target:
                    target[key] = []
                target[key].extend(item)
            else: 
                if key not in target:
                    target[key] = []
                target[key].append(item)


def extend_dict(source: Dict[str, Any], target: Dict[str, Any], 
                include_keys: List[str] = []):
    """Extends element in target dictionary by each elements in source-dict.
    
    Assumes source-dict has as element as list or again a dict. Only keys in
    include_keys are considered. If a sub-dictionary is included, all its keys
    will be included.
    """
    for key, item in source.items():
        if key in include_keys:
            if isinstance(item, dict):
                if key not in target:
                    target[key] = {}
                extend_dict(item, target[key], include_keys = item.keys())
            else: 
                if key not in target:
                    target[key] = []
                target[key].extend(item)


def average_dict(target: Dict[str, Any]):
    """Recursively averages every value-holding element in target-dictionary. 
    
    Assumes target dictionary holds str-keys and either list or dict items. 
    Every list-item is averaged. If element is a dict, assumes it is 
    shallow only containing lists.
    """
    keys = [key for key in target]
    for key in keys:
        item = target[key]
        if isinstance(item, dict):
            avg_dict, std_dict, sem_dict = average_subdict(item)
            target["sem_" + key] = sem_dict
            target["avg_" + key] = avg_dict
            target["std_" + key] = std_dict
        else:
            assert len(target[key]) > 0
            target["sem_" + key] = scipy.stats.sem(item, ddof=0)
            target["avg_" + key] = np.mean(item)
            target["std_" + key] = np.std(item)


def average_subdict(subdict: Dict[str, Any]):
    """Return a dictionary with averaged and a dictionary with standard deviation
    values and standard error of the mean for each element."""
    keys = [key for key in subdict]
    avg_dict = {}
    std_dict = {}
    sem_dict = {}
    for key in keys:
        item = subdict[key]
        assert not isinstance(item, dict)
        assert len(subdict[key]) > 0
        avg_dict[key] = np.mean(item)
        std_dict[key] = np.std(item)
        sem_dict[key] = scipy.stats.sem(item, ddof=0)
    return avg_dict, std_dict, sem_dict


class Experiment:
    """An experiment refers to the (robustness) results optained by a 
    particular model on K averaged over multiple seeds."""
    def __init__(self, experiment_list: List[Dict[str, Any]]):
        assert len(experiment_list) > 0
        self.individual_experiments = experiment_list
        self.id = experiment_list[0]["_id"]
        self.hyperparameters = experiment_list[0]["config"]
        self.label = self.hyperparameters["model_params"]["label"]
        self.C = float(self.hyperparameters["model_params"]["regularizer"])
        if "certificate_params" in self.hyperparameters:
            self.attack = False
            params = self.hyperparameters["certificate_params"]
        else:
            self.attack = True
            params = self.hyperparameters["attack_params"]
        self.delta = float(params["delta"])
        # self.n_adv = int(params["n_adversarial"])
        # self.attack_nodes = params["attack_nodes"]
        Experiment.assert_same_hyperparameters(self.individual_experiments)
        self.average_result_statistics()

    @staticmethod
    def assert_same_hyperparameters(
        individual_experiments: List[Dict[str, Any]]
    ) -> None:
        """Sanity check if all given experiments indeed have the same 
        configuration."""
        data_params_l = []
        model_params_l = []
        certificate_params_l = []
        for experiment in individual_experiments:
            data_params = experiment["config"]["data_params"]
            del data_params["specification"]["seed"]
            data_params_l.append(experiment["config"]["data_params"])
            model_params_l.append(experiment["config"]["model_params"])
            if "certificate_params" in experiment["config"]:
                certificate_params_l.append(experiment["config"]["certificate_params"])
            else:
                certificate_params_l.append(experiment["config"]["attack_params"])
        for i in range(1, len(individual_experiments)):
            assert_equal_dicts(data_params_l[0], data_params_l[i])
            assert_equal_dicts(data_params_l[0], data_params_l[i])
            assert_equal_dicts(model_params_l[0], model_params_l[i])
    
    def average_result_statistics(self):
        """Average prediction statistics and robustness statistics calculated 
        from the raw data for each seed."""
        self.results = {}
        for experiment in self.individual_experiments:
            if "result" not in experiment:
                print(f"{experiment['_id']} {self.label} delta: {self.delta} has no result.")
                continue
            result = experiment["result"]
            if result is not None:
                append_dict(result, self.results)
            else:
                print(f"{experiment['_id']} {self.label} delta: {self.delta} is NONE.")
        average_dict(self.results)

    def get_result(self, key: str) -> Tuple[float, float]:
        """Return average & std of metric 'key' over all seeds."""
        return np.mean(self.results[key]).item(), np.std(self.results[key]).item()
    
    def get_robust_accuracy(self) -> Tuple[float, float]:
        """
        For certificates returns certified accuracy.
        For attacks returns roubst accuracy.
        """
        n_robust_acc_l = []
        self.n_test = self.hyperparameters["data_params"]["specification"]["n_test"]
        for experiment in self.individual_experiments:
            n_robust_acc = 0
            result = experiment["result"]
            for y_true, y_pred, y_worst in zip(result["y_true_cls"],
                                               result["y_pred_logit"],
                                               result["y_worst_obj"]):
                if y_pred > 0 and y_true > 0 and y_worst > 0:
                    n_robust_acc += 1
                if y_pred < 0 and y_true < 0 and y_worst < 0:
                    n_robust_acc += 1
            n_robust_acc_l.append(n_robust_acc / self.n_test)
        return np.mean(n_robust_acc_l).item(), np.std(n_robust_acc_l).item()
    
    def get_certified_ratio(self) -> Tuple[float, float]:
        """
        For certificates returns certified robustness (independent of correct class.)
        For attacks returns robust predictions (independent of correct class.)
        """
        n_robust_l = []
        self.n_test = self.hyperparameters["data_params"]["specification"]["n_test"]
        for experiment in self.individual_experiments:
            result = experiment["result"]
            n_robust = 0
            for y_true, y_pred, y_worst in zip(result["y_true_cls"],
                                               result["y_pred_logit"],
                                               result["y_worst_obj"]):
                if y_pred > 0 and y_worst > 0:
                    n_robust += 1
                if y_pred < 0 and y_worst < 0:
                    n_robust += 1
            n_robust_l.append(n_robust / self.n_test)
        return np.mean(n_robust_l).item(), np.std(n_robust_l).item()
    
    def __str__(self):
        my_str = self.label
        my_str += f" K: {self.K:.1f}, C: {self.C:.5f}, delta: {self.delta:.2f},"
        my_str += f" n_adv: {self.n_adv}, attack_nodes: {self.attack_nodes}"
        return my_str


def get_robust_accuracy(exp: Experiment) -> Tuple[float, float]:
    """
    For certificates returns certified accuracy.
    For attacks returns roubst accuracy.
    """
    n_robust_acc_l = []
    for experiment in exp.individual_experiments:
        n_robust_acc = 0
        result = experiment["result"]
        if result is None:
            continue
        n_test = len(result["y_true_cls"])
        for y_true, y_pred, y_robust in zip(result["y_true_cls"],
                                            result["y_pred_logit"],
                                            result["y_is_robust"]):
            if y_pred > 0 and y_true > 0 and y_robust > 1e-4:
                n_robust_acc += 1
            if y_pred < 0 and y_true < 0 and y_robust > 1e-4:
                n_robust_acc += 1
        n_robust_acc_l.append(n_robust_acc / n_test)
    return np.mean(n_robust_acc_l).item(), np.std(n_robust_acc_l).item()


def get_certified_ratio(exp: Experiment) -> Tuple[float, float]:
    """
    For certificates returns certified ratio.
    For attacks returns roubst ratio.
    """
    n_robust_rat_l = []
    for experiment in exp.individual_experiments:
        n_robust_acc = 0
        result = experiment["result"]
        if result is None:
            continue
        n_test = len(result["y_true_cls"])
        y_robust_c = np.sum(result["y_is_robust"])
        n_robust_rat_l.append(y_robust_c / n_test)
    return np.mean(n_robust_rat_l).item(), np.std(n_robust_rat_l).item()


def get_experiment(exp: Experiment, seed=0):
    for ind_exp in exp.individual_experiments:
        if ind_exp["config"]["seed"] == seed:
            idx_labeled = ind_exp["result"]["idx_labeled"]
            idx_test = ind_exp["result"]["idx_test"]
            y_flip = ind_exp["result"]["y_flip"]
            y_is_robust = ind_exp["result"]["y_is_robust"]
            print(idx_labeled, idx_test, y_flip, y_is_robust)
    return idx_labeled, idx_test, y_flip, y_is_robust


class ExperimentManager:
    """Administrates access and visualization of robustness experiments.
    
    Assumes same experiments with different seeds are stored consecutively.
    """
    def __init__(self, experiments: List[Dict[str, Any]], uri=URI):
        """Establish connection to a given mongodb collection. 
        
        Load and administrate data from specified experiments.
        """
        self.client = MongoClient(uri)
        self.db = self.client.sabanaya
        self.load(experiments)
        self.init_plot_params()

    def load_experiment_dict(self, id: int, collection: str) -> Dict[str, Any]:
        """Return result-dict of experiment with ID id."""
        return self.db[collection].find_one({'_id': id})

    def load_experiment(self, start_id: int, n_seeds: int, 
                        collection: str) -> Experiment:
        """Return experiment with ID id."""
        exp_dict_l = []
        for i in range(n_seeds):
            exp_dict_l.append(self.load_experiment_dict(start_id + i, 
                                                        collection))
        return Experiment(exp_dict_l)

    def load_experiments(
            self, start_id: int, end_id: int, n_seeds: int, label: str=None,
            collection: str="runs",
        ) -> List[Experiment]:
        """Return Experiments between start_id and end_id with given label.
        
        Assumes that one experiment consists of multiple seeds which are stored
        consecutively in the mongodb in a given collection.
        """
        experiment_ids = [i for i in range(start_id, end_id + 1, n_seeds)]
        experiments = [self.load_experiment(i, n_seeds, collection) 
                       for i in experiment_ids]
        if label is not None:
            filtered_experiments = []
            for experiment in experiments:
                if experiment.label == label:
                    filtered_experiments.append(experiment)
            print(f'{label} No. of experiments {len(filtered_experiments)}')
            return filtered_experiments
        else:
            return experiments

    def load(self, experiments) -> None:
        """Populates experiments_dict from stored results in MongoDB.
        
        Experiment_dict is populated as a two-level dictionary. First level
        has the label of the experiment as key and second-level the K.
        """
        self.experiments_dict = {}
        for exp_spec in experiments:
            if "collection" not in exp_spec:
                exp_spec["collection"] = "runs"
            if not isinstance(exp_spec["start_id"], Iterable):
                exp_spec["start_id"] = [exp_spec["start_id"]]
                exp_spec["end_id"] = [exp_spec["end_id"]]
            for start_id, end_id in zip(exp_spec["start_id"], exp_spec["end_id"]):
                if "label" in exp_spec:
                    label = exp_spec["label"]
                else:
                    label = None
                exp_list = self.load_experiments(start_id,
                                                 end_id,
                                                 exp_spec["n_seeds"],
                                                 label,
                                                 exp_spec["collection"])
                for exp in exp_list:
                    if "postfix" in exp_spec:
                        exp.label += exp_spec["postfix"]
                    if "relabel" in exp_spec:
                        if exp_spec["relabel"] not in self.experiments_dict:
                            self.experiments_dict[exp_spec["relabel"]] = {}
                    elif exp.label not in self.experiments_dict:
                        self.experiments_dict[exp.label] = {}
                    label = exp_spec["relabel"] if "relabel" in exp_spec else exp.label
                    # if exp.attack_nodes not in self.experiments_dict[label][exp.K][exp.C]:
                    #     self.experiments_dict[label][exp.K][exp.C][exp.attack_nodes] = {}
                    # if exp.n_adv not in self.experiments_dict[label][exp.K][exp.C][exp.attack_nodes]:
                    #     self.experiments_dict[label][exp.K][exp.C][exp.attack_nodes][exp.n_adv] = {}
                    self.experiments_dict[label][exp.delta] = exp

    def init_plot_params(self):
        # Matplotlib settings
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['mathtext.rm'] = 'serif'
        mpl.rcParams['savefig.dpi'] = 600
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 16
        mpl.rcParams['axes.formatter.use_mathtext'] = True
        mpl.rcParams['text.usetex'] = True

        mpl.rcParams['font.family'] = 'STIXGeneral'
        mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

        mpl.rcParams['legend.fontsize'] = 12
        # mpl.rcParams['legend.labelspacing'] = 0.3
        # mpl.rcParams['legend.borderpad'] = 0.2
        # mpl.rcParams['legend.handlelength'] = 1

    def get_style(self, label: str):
        
        """
        color_dict = {
            "APPNP_alpha1": 'slategrey', #MLP
            "MLP": 'slategrey', #MLP
            "GCN": 'tab:green', 
            "GCN_sym": 'tab:green', 
            "APPNP_alpha0": "plum",
            "APPNP_alpha0.1": "tab:brown",
            "APPNP_alpha0.1_row": "tab:brown",
            "APPNP_alpha0.2": "r",
            "APPNP": 'r', #lime 
            "APPNP_alpha0.3": "tab:olive",
            "APPNP_alpha0.3_row": "tab:olive",
            "APPNP_alpha0.5": "darkslategrey",
            "SGC": "blue",
            "SGC_sym": "blue",
            "GCN_skippc": "lime", #k
            "GCN_skippc_linear": "lime", #k
            "GCN_skippc_relu+2": "lime",
            "GCN_skipalpha": "plum", #"wheat",
            "GCN_skipalpha_linear_alpha0.2": "wheat",
            "GCN_skipalpha_relu_alpha0.2+2": "wheat",
            "GCN_skipalpha_linear_alpha0.1": "steelblue",
            "GCN_skipalpha_relu_alpha0.1+2": "steelblue",
            "GIN": "darkslateblue",
            "GraphSAGE": "darkred",
            # "GAT": "slategrey",
            # "GATv2": "k",
            # "GraphSAGE": "lightsteelblue",
            # "LP": "wheat",
        }"""
        color_dict = {
            "GCN": "black",
            "GCN_csbm": "black",
            "GCN_coramlb": "mediumseagreen",
            "GCN_citeseerb": "fuchsia",
            "GCN_L2": "black",
            "GCN_L4": "black",
            "GCN_sym": "black",
            "APPNP": "lime",
            "APPNP_alpha0.1": "lime",
            "APPNP_alpha0.2": "lime",
            "APPNP_alpha0.3": "lime",
            "APPNP_alpha0.5": "lime",
            "SGC": "fuchsia",
            "SGC_csbm": "black",
            "SGC_coramlb": "mediumseagreen",
            "SGC_citeseerb": "fuchsia",
            "SGC_sym": "fuchsia",
            "GCN_skippc": "darkslateblue",
            "GCN_skippc_L2": "darkslateblue",
            "GCN_skippc_L4": "darkslateblue",
            "GCN_skipalpha_L2": "deepskyblue",
            "GCN_skipalpha_L4": "deepskyblue",
            "GCN_skipalpha": "deepskyblue",
            "GraphSAGE": "mediumseagreen",
            "GIN": "saddlebrown",
            "LinearKernel": "tab:olive",
            "MLP": "slategrey",
        }
        linestyle_dict = {
            "LP": '--',
            "GCN_csbm": ":",
            "GCN_coramlb": ":",
            "GCN_citeseerb": ":",
            "SGC_sym": ":",
            "GCN_sym": ":",
            "APPNP_alpha0.1": (0, (3, 5, 1, 5)),
            "APPNP_alpha0.2": (0, (5, 1)),
            "APPNP_alpha0.3": "dotted",
            "APPNP_alpha0.5": "densely dotted",
            "APPNP_alpha0.1_row": "dashed",
            "APPNP_alpha0.3_row": "dashed",
            "MLP": 'dashed',
            "LinearKernel": 'dashed',
        }
        use_color=""
        linestyle="-"
        for key, color in color_dict.items():
            _label = label
            if label.endswith("_coll") or label.endswith("_samp"):
                _label = label[:-5]
            if key == _label:
                use_color = color
        for key, style in linestyle_dict.items():
            if key == label:
                linestyle = style
            elif label.endswith("_samp"):
                linestyle = (0, (1,1))
        return use_color, linestyle
    
    def set_color_cycler(self, ax):
        color_list = ['fuchsia', 
                      'darkslateblue', 
                      'deepskyblue', 
                      'lime', 
                      'slategrey', 
                      'black', 
                      "mediumseagreen",
                      "saddlebrown",
                      ]
        linestyle_list = ['-', '--', ':', '-.']
        ax.set_prop_cycle(cycler('linestyle', linestyle_list)*
                          cycler('color', color_list))

    def set_xaxis_labels(self, ax, x_ticks, x_labels, fontsize):
        ax.xaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_ticks(x_ticks, minor=False)
        xticks = [f"{label}" for label in x_labels]
        ax.xaxis.set_ticklabels(xticks, fontsize=fontsize, fontweight="bold")
        #ax.set_xlim(left=-0.3)
    
    def set_xaxis_labels_logscale(self, ax, x_ticks, x_labels):
        ax.xaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_ticks(x_ticks, minor=False)
        xticks = [f"{label}" for label in x_labels]
        ax.xaxis.set_ticklabels(xticks)

    def plot_robust_auc(self, models: List[str], 
                              delta_l: List[float],
                              legend_labels: List[str]=[],
                              width=1, ratio=1.618, 
                              scale=True,
                              barwidth=1,
                              savefig: str=None,
                              savedir: Path=None):
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        # self.set_color_cycler(ax)
        if len(legend_labels) != len(models):
            legend_labels = models
        aoc_l = []
        aoc_lb = []
        aoc_ub = []
        for (label, legend_label) in zip(models, legend_labels):
            y_err_l = []
            y_l = []
            for delta in delta_l:
                if delta == 0.:
                    exp = self.experiments_dict[label][delta_l[1]]
                    y, y_std = exp.get_result("accuracy_test")
                else:
                    exp = self.experiments_dict[label][delta]
                    y, y_std = get_robust_accuracy(exp)
                y_l.append(y)
                y_err_l.append(y_std)
            x = np.array(delta_l)
            aoc_l.append(np.trapz(y_l, x))
            aoc_lb.append(np.trapz(np.array(y_l) - np.array(y_err_l)/np.sqrt(10), x))
            aoc_ub.append(np.trapz(np.array(y_l) + np.array(y_err_l)/np.sqrt(10), x))
        # scale by most robust model
        if scale:
            aoc_max = np.max(aoc_l)
            aoc_l = np.array(aoc_l) / np.max(aoc_l)
            idx_sorted = np.argsort(aoc_l)[::-1]
            aoc_lb = np.array(aoc_lb) / aoc_max
            aoc_ub = np.array(aoc_ub) / aoc_max
            x = np.arange(len(aoc_l))
            ax.set_ylim(bottom=0.75, top=1.05)
            ax.set_xticks(x)
            ax.set_xticklabels([legend_labels[i] for i in idx_sorted])
            ax.bar(x, aoc_l[idx_sorted], 
                   yerr=[aoc_l[idx_sorted] - aoc_lb[idx_sorted], 
                        aoc_ub[idx_sorted] - aoc_l[idx_sorted]], capsize=3,
                   width=barwidth)
            ax.set_xticks(x)
            ax.set_xticklabels([legend_labels[i] for i in idx_sorted])
        else:
            x = np.arange(len(aoc_l))
            ax.bar(x, np.array(aoc_l), 
                yerr=[np.array(aoc_l) - np.array(aoc_lb), 
                      np.array(aoc_ub) - np.array(aoc_l)], capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(legend_labels)
        if savefig:
            if savedir is None:
                savedir = CERTIFICATE_FIGURE_DIR
            savedir.mkdir(parents=True, exist_ok=True)
            plt.savefig(savedir/savefig, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def plot_robust_acc_delta(self, models: List[str], 
                              delta_l: List[float],
                              legend_labels: List[str]=[],
                              certified_ratio: bool=False,
                              width=1, ratio=1.618, 
                              xlogscale: bool=False,
                              savefig: str=None,
                              use_style: bool=True,
                              legend_y: str="",
                              legend_x: str="",
                              label_fontsize=16,
                              legend_fontsize=12,
                              ticks_fontsize=10,
                              y_lim_ticks=1,
                              markersize=4,
                              capsize=3,
                              linewidth=1,
                              framealpha=1.0,
                              use_custom_legend=False,
                              legend_args=None,
                              plot_coll_and_sample=False):
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        if not use_style:
            self.set_color_cycler(ax)
        if len(legend_labels) != len(models):
            legend_labels = models
        for (label, legend_label) in zip(models, legend_labels):
            y_err_l = []
            y_l = []
            for delta in delta_l:
                if delta == 0.:
                    exp = self.experiments_dict[label][delta_l[1]]
                    if certified_ratio:
                        y, y_std = 1, 0
                    else:
                        y, y_std = exp.get_result("accuracy_test")
                else:
                    exp = self.experiments_dict[label][delta]
                    if certified_ratio:
                        y, y_std = get_certified_ratio(exp)
                    else:
                        y, y_std = get_robust_accuracy(exp)
                y_l.append(y)
                y_err_l.append(y_std)
            if xlogscale:
                ax.set_xscale('log')
                x = np.array(delta_l)
                if x[0] == 0:
                    x[0] = 0.005
                self.set_xaxis_labels_logscale(ax, x, delta_l)
            else:
                x = [i for i in range(len(delta_l))]
                self.set_xaxis_labels(ax, x, delta_l, ticks_fontsize)
            
            label_str = r'{0}'.format(legend_label) #+ " " + str(C)
            if use_style:
                color, linestyle = self.get_style(label)
                ax.errorbar(x, y_l, yerr=y_err_l, marker="o", label=label_str, 
                            color=color, linestyle=linestyle,
                            capsize=capsize, linewidth=linewidth, 
                            markersize=markersize)
            else:
                    ax.errorbar(x, y_l, yerr=y_err_l, marker="o", label=label_str, 
                            capsize=capsize, linewidth=linewidth, 
                            markersize=markersize)

        if use_custom_legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            newLabels, newHandles = [], []
            for handle, label in zip(handles, labels):
                if label not in newLabels:
                    newLabels.append(label)
                    newHandles.append(handle)
            if plot_coll_and_sample:
                if len(newLabels) % 2 == 1:
                    newLabels.append("Sample")
                    newLabels.append("Collective")
                    newHandles.append(plt.Line2D([0], [0], color='black', linestyle=(0, (1,1))))
                    newHandles.append(plt.Line2D([0], [0], color='black', linestyle='-'))
                if len(newLabels) % 2 == 0:
                    idx = int(len(newLabels) / 2)
                    newLabels.insert(idx, "Sample")
                    newLabels.append("Collective")
                    newHandles.insert(idx, plt.Line2D([0], [0], color='black', linestyle=(0, (1,1))))
                    newHandles.append(plt.Line2D([0], [0], color='black', linestyle='-'))

            ax.legend(newHandles,newLabels,
                    fontsize=legend_fontsize, 
                    framealpha=legend_args["framealpha"],
                    handlelength=legend_args["handlelength"],
                    handletextpad=legend_args["handletextpad"],
                    labelspacing = legend_args["labelspacing"], 
                    loc=legend_args["loc"],
                    ncol=legend_args["ncol"],
                    columnspacing=legend_args["columnspacing"],
                    bbox_to_anchor=legend_args["bbox_to_anchor"])
        else:
            ax.legend(fontsize=legend_fontsize, framealpha=framealpha)
        ax.set_ylabel(legend_y, fontsize=label_fontsize)
        y_ticks_minor = np.arange(0, y_lim_ticks+0.01, 0.05)
        ax.set_yticks(y_ticks_minor, minor=True)
        ax.set_xlabel(legend_x, fontsize=label_fontsize)
        ax.tick_params(labelsize=ticks_fontsize)
        ax.yaxis.grid()
        ax.yaxis.grid(which='minor', alpha=0.3)
        ax.xaxis.grid()
        if savefig:
            CERTIFICATE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(CERTIFICATE_FIGURE_DIR/savefig, bbox_inches='tight')
        plt.show()

    def plot_certified_ratio_delta(self, models: List[str], 
                              delta_l: List[float],
                              legend_labels: List[str]=[],
                              width=1, ratio=1.618, 
                              xlogscale: bool=False,
                              savefig: str=None,
                              use_style: bool=True):
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        if not use_style:
            self.set_color_cycler(ax)
        if len(legend_labels) != len(models):
            legend_labels = models
        for (label, legend_label) in zip(models, legend_labels):
            y_err_l = []
            y_l = []
            for delta in delta_l:
                if delta == 0.:
                    exp = self.experiments_dict[label][delta_l[1]]
                    y, y_std = 1., 0.
                else:
                    exp = self.experiments_dict[label][delta]
                    y, y_std = get_certified_ratio(exp)
                    print(f"delta: {delta:.2f} racc: {y:.3f} std: {y_std:.2f}")
                y_l.append(y)
                y_err_l.append(y_std)
            if xlogscale:
                ax.set_xscale('log')
                x = np.array(delta_l)
                if x[0] == 0:
                    x[0] = 0.005
                self.set_xaxis_labels_logscale(ax, x, delta_l)
            else:
                x = [i for i in range(len(delta_l))]
                self.set_xaxis_labels(ax, x, delta_l)
            
            label_str = r'{0}'.format(legend_label) #+ " " + str(C)
            if use_style:
                color, linestyle = self.get_style(label)
                ax.errorbar(x, y_l, yerr=y_err_l, marker="o", label=label_str, 
                            color=color, linestyle=linestyle,
                            capsize=3, linewidth=1, markersize=4)
            else:
                    ax.errorbar(x, y_l, yerr=y_err_l, marker="o", label=label_str, 
                                capsize=3, linewidth=1, markersize=4)

        ax.set_ylabel("Certified Ratio")
        ax.set_xlabel(r"Perturbation budget $\delta$")
        ax.yaxis.grid()
        ax.xaxis.grid()
        ax.legend()
        if savefig:
            CERTIFICATE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(CERTIFICATE_FIGURE_DIR/savefig, bbox_inches='tight')
        plt.show()

    def rank_models(self, models: List[str],
                    delta_l: List[float],
                    legend_labels: List[str]=[],
                    weak_threshold = 0.1,
                    intermediate_threshold = 0.3,
                    decimal_round_off = 2,
                    include_model_rank = True,
                    ):
        if legend_labels == []:
            legend_labels = models
        model_acc = []
        model_acc_std = []
        for m_id in range(len(models)):
            label = models[m_id]
            y_err_l = []
            y_l = []
            for delta in delta_l:
                if delta == 0.:
                    exp = self.experiments_dict[label][delta_l[1]]
                    y, y_std = 1, 0
                else:
                    exp = self.experiments_dict[label][delta]
                    y, y_std = get_certified_ratio(exp)
                y_l.append(y)
                y_err_l.append(y_std)
            model_acc.append(y_l)
            model_acc_std.append(y_err_l)
        model_acc = np.array(model_acc)
        model_acc_std = np.array(model_acc_std)
        # print(model_acc)
        model_rank = rankdata(-model_acc, axis=0, method='min')
        w_id = delta_l.index(weak_threshold)+1
        i_id = delta_l.index(intermediate_threshold)+1
        weak = np.array([np.mean(model_rank[:,:w_id], axis=1)]).round(decimals=decimal_round_off)
        intermediate = np.array([np.mean(model_rank[:,w_id:i_id], axis=1)]).round(decimals=decimal_round_off)
        strong = np.array([np.mean(model_rank[:,i_id:], axis=1)]).round(decimals=decimal_round_off)
        total = np.array([np.mean(model_rank, axis=1)]).round(decimals=decimal_round_off)
        labels = np.array([legend_labels])
        if include_model_rank:
            df_header = ["model"] + delta_l + ["weak", "intermediate", "strong", "total"]
            df_data = np.concatenate((labels.T, model_rank, weak.T, intermediate.T, strong.T, total.T), axis=1)
        else:
            df_header = ["model"] + ["weak", "intermediate", "strong", "total"]
            df_data = np.concatenate((labels.T, weak.T, intermediate.T, strong.T, total.T), axis=1)
        df = pd.DataFrame(data = df_data, columns = df_header)
        return df

    def models_delta(self, models: List[str],
                    delta_l: List[float],
                    legend_labels: List[str]=[],
                    bold = True,
                    certified_ratio = True,
                    ):
        if legend_labels == []:
            legend_labels = models
        model_acc = []
        model_acc_std = []
        for m_id in range(len(models)):
            label = models[m_id]
            y_err_l = []
            y_l = []
            for delta in delta_l:
                if delta == 0.:
                    exp = self.experiments_dict[label][delta_l[1]]
                    if certified_ratio:
                        y, y_std = 1, 0
                    else:
                        y, y_std = exp.get_result("accuracy_test")
                else:
                    exp = self.experiments_dict[label][delta]
                    if certified_ratio:
                        y, y_std = get_certified_ratio(exp)
                    else:
                        y, y_std = get_robust_accuracy(exp)
                    
                y_l.append(y)
                y_err_l.append(y_std)
            model_acc.append(y_l)
            model_acc_std.append(y_err_l)
        model_acc = np.array(model_acc) * 100
        model_acc_std = np.array(model_acc_std) * 100
        model_rel_acc = model_acc - [model_acc[0]]
        model_rel_acc[0,:] = model_acc[0,:]
        model_str_l = []

        i = 0
        max_indices = np.argmax(model_rel_acc[1:], axis=0) + 1
        min_indices = np.argmin(model_rel_acc[1:], axis=0) + 1
        for acc_line, std_line in zip(model_rel_acc, model_acc_std):
            acc_l = []
            j = 0
            for acc, std, max_idx, min_idx in zip(acc_line, std_line, max_indices, min_indices):
                if acc >= 0 and i > 0:
                    if max_idx == i:
                        acc_l = acc_l + [f"\\textbf{{+{acc:.1f}}} \\cellcolor{{gray!30}}\scriptsize{{$\mathbf{{\pm}}$\\textbf{{{std:.1f}}}}}"]
                    elif min_idx == i:
                        acc_l = acc_l + [f"\\textcolor{{red}} {acc:.1f}\scriptsize{{$\pm$ {std:.1f}}}"]
                    else:
                        acc_l = acc_l + [f"+{acc:.1f} \scriptsize{{$\pm$ {std:.1f}}}"]
                else:
                    acc_l = acc_l + [f"{acc:.1f} \scriptsize{{$\pm$ {std:.1f}}}"]
                j += 1
            i += 1
            model_str_l.append(acc_l)
        labels = np.array([legend_labels])
        if bold:
            delta_l = [f"$\mathbf{{{delta:.2f}}}$" for delta in delta_l]
            labels = np.array([[f"\\textbf{{{label}}}" for label in labels[0]]])
        df_header = ["$\\epsilon$"] + delta_l
        df_data = np.concatenate((labels.T, model_str_l), axis=1)
        df = pd.DataFrame(data = df_data, columns = df_header)
        return df
    
    def models_robustness(self, models: List[str],
                    delta_l: List[float],
                    legend_labels: List[str]=[],
                    bold = True,
                    certified_ratio = False,
                    to_latex = True,
                    ):
        if legend_labels == []:
            legend_labels = models
        model_acc = []
        model_acc_std = []
        for m_id in range(len(models)):
            label = models[m_id]
            y_err_l = []
            y_l = []
            for delta in delta_l:
                if delta == 0.:
                    exp = self.experiments_dict[label][delta_l[1]]
                    if certified_ratio:
                        y, y_std = 1, 0
                    else:
                        y, y_std = exp.get_result("accuracy_test")
                else:
                    exp = self.experiments_dict[label][delta]
                    if certified_ratio:
                        y, y_std = get_certified_ratio(exp)
                    else:
                        y, y_std = get_robust_accuracy(exp)
                y_l.append(y)
                y_err_l.append(y_std)
            model_acc.append(y_l)
            model_acc_std.append(y_err_l)
        model_acc = np.array(model_acc) * 100
        model_acc_std = np.array(model_acc_std) * 100

        model_str_l = []
        i = 0
        max_indices = np.argmax(model_acc, axis=0)
        for acc_line, std_line in zip(model_acc, model_acc_std):
            acc_l = []
            j = 0
            for acc, std, max_idx in zip(acc_line, std_line, max_indices):
                if to_latex:
                    if max_idx == i:
                        acc_l = acc_l + [f"\\textbf{{{acc:.1f}}}$\mathbf{{\pm}}$\\textbf{{{std:.1f}}}"]
                    else:
                        acc_l = acc_l + [f"{acc:.1f} $\pm$ {std:.1f}"]
                else:
                    acc_l = acc_l + [f"{acc:.1f}±{std:.1f}"]
                j += 1
            i += 1
            model_str_l.append(acc_l)
        labels = np.array([legend_labels])
        if bold and to_latex:
            delta_l = [f"$\mathbf{{{delta:.2f}}}$" for delta in delta_l]
            labels = np.array([[f"\\textbf{{{label}}}" for label in labels[0]]])
        df_header = ["$\\epsilon$"] + delta_l
        df_data = np.concatenate((labels.T, model_str_l), axis=1)
        df = pd.DataFrame(data = df_data, columns = df_header)
        return df

    def plot_robustness_delta(self, models: List[str],
                    delta_l: List[float],
                    legend_labels: List[str]=[],
                    bold = True,
                    certified_ratio = False,
                    to_latex = True,
                    ):
        if legend_labels == []:
            legend_labels = models
        for m_id in range(len(models)):
            label = models[m_id]
            exp = self.experiments_dict[label][0.1]
            y_01, y_std = get_certified_ratio(exp) 
            exp = self.experiments_dict[label][0.3]
            y_03, y_std = get_certified_ratio(exp) 
            exp = self.experiments_dict[label][0.5]
            y_05, y_std = get_certified_ratio(exp) 
            y_01 = y_01*100
            y_03 = y_03*100
            y_05 = y_05*100
            print(f"Model {label} has delta_m: {y_01-y_03:.2f} delta_s: {y_03-y_05:.2f}")
                    
    def plot_robust_acc_delta_nadv(self, K: float, models: List[str], C_l: List[float], 
                              attack_nodes: str, n_adv_l: List[int], delta_l: List[float],
                              width=1, ratio=1.618):
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        self.set_color_cycler(ax)
        for label in models:
            for C in C_l:
                for n_adv in n_adv_l:
                    y_err_l = []
                    y_l = []
                    for delta in delta_l:
                        if delta == 0.:
                            exp = self.experiments_dict[label][K][C][attack_nodes][n_adv][delta_l[1]]
                            y, y_std = exp.get_result("accuracy_test")
                        else:
                            exp = self.experiments_dict[label][K][C][attack_nodes][n_adv][delta]
                            y, y_std = exp.get_robust_accuracy()
                        y_l.append(y)
                        y_err_l.append(y_std)
                    x = [i for i in range(len(delta_l))]
                    label_str = label + " " + str(C) + f" n_adv {n_adv}"
                    ax.errorbar(x, y_l, yerr=y_err_l, marker="o", label=label_str, 
                                capsize=3, linewidth=1, markersize=4)
                    self.set_xaxis_labels(ax, x, delta_l)
        ax.set_ylabel("Robust Accuracy", fontsize=20)
        ax.set_xlabel(r"$\delta$", fontsize=17, fontweight="bold")
        ax.yaxis.grid()
        ax.xaxis.grid()
        ax.legend()
        plt.show()

    def plot_nadv_delta_robust_gain_wrt_mlp_heatmap(self, K: float, models: str, C: float, 
                              attack_nodes: str, n_adv_l: List[str], delta_l: List[float], mlp: str,
                              title_label: str=None, width=1, ratio=1.2, savefig: str=None):
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        # self.set_color_cycler(ax)
        for label in models:
            nadv_delta = np.zeros((len(delta_l), len(n_adv_l)))
            for i in range(len(delta_l)):
                for j in range(len(n_adv_l)):
                    delta = delta_l[i]
                    n_adv = int(n_adv_l[j])
                    if delta == 0:
                        exp = self.experiments_dict[label][K][C][attack_nodes][n_adv][0.01]
                        acc_exp, _ = exp.get_result("accuracy_test")
                        mlp_exp = self.experiments_dict[mlp][K][C][attack_nodes][n_adv][0.01]
                        acc_mlp, _ = mlp_exp.get_result("accuracy_test")
                        acc_diff = acc_exp-acc_mlp
                    else:
                        exp = self.experiments_dict[label][K][C][attack_nodes][n_adv][delta]
                        acc_exp, _ = exp.get_robust_accuracy()
                        mlp_exp = self.experiments_dict[mlp][K][C][attack_nodes][n_adv][0.01]
                        acc_mlp, _ = mlp_exp.get_robust_accuracy()
                        acc_diff = acc_exp-acc_mlp
                    nadv_delta[i][j] = acc_diff
        cmap = matplotlib.cm.get_cmap('coolwarm_r')
        sns.heatmap(nadv_delta, cmap=cmap, center=0, linewidths=0.5, cbar=True, 
                    cbar_kws={'label': 'Certified accuracy gain'})
        ax.set_xticks(np.arange(nadv_delta.shape[1])+0.5, labels=n_adv_l)
        ax.set_yticks(np.arange(nadv_delta.shape[0])+0.5, labels=delta_l, rotation=0)
        ax.set_ylabel(r"Perturbation budget $\delta$")
        ax.set_xlabel("Number of adversaries")
        if title_label:
            ax.set_title(title_label[0])
        else:
            ax.set_title(models[0])
        if savefig:
            CERTIFICATE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(CERTIFICATE_FIGURE_DIR/savefig, bbox_inches='tight', dpi=600)
        plt.show()

    def plot_nadv_delta_heatmap(self, K: float, models: str, C: float, 
                              attack_nodes: str, n_adv_l: List[str], delta_l: List[float],
                              width=1, ratio=1.618, cbar_normalized = True):
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        self.set_color_cycler(ax)
        
        for label in models:
            nadv_delta = np.zeros((len(delta_l), len(n_adv_l)))
            for i in range(len(delta_l)):
                for j in range(len(n_adv_l)):
                    delta = delta_l[i]
                    n_adv = int(n_adv_l[j])
                    if delta == 0:
                        #exp = self.experiments_dict[label][K][C][attack_nodes][n_adv][0.01]
                        #y, y_std = exp.get_result("accuracy_test")
                        y = 1
                    else:
                        exp = self.experiments_dict[label][K][C][attack_nodes][n_adv][delta]
                        y, y_std = exp.get_certified_ratio()
                    nadv_delta[i][j] = y
        cmap = matplotlib.cm.get_cmap('Greys')
        if cbar_normalized:
            sns.heatmap(nadv_delta, cmap=cmap, linewidths=0.5, cbar=True, 
                    cbar_kws={'label': 'Certified ratio'}, vmin=0, vmax=1)
        else:
            sns.heatmap(nadv_delta, cmap=cmap, linewidths=0.5, cbar=True, 
                    cbar_kws={'label': 'Certified ratio'})
        ax.set_xticks(np.arange(nadv_delta.shape[1])+0.5, labels=n_adv_l)
        ax.set_yticks(np.arange(nadv_delta.shape[0])+0.5, labels=delta_l, rotation=0)
        ax.set_ylabel("Perturbation budget")
        ax.set_xlabel("Number of adversaries")
        ax.set_title(models[0])
        plt.show()

    def plot_collective_cert_delta(self, K: float, models: List[str], C_l: List[float], 
                              attack_nodes: str, n_adv: int, delta_l: List[float],
                              legend_labels: List[str]=[],
                              width=1, ratio=1.618, 
                              xlogscale: bool=False,
                              savefig: str=None):
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        self.set_color_cycler(ax)
        if len(legend_labels) != len(models):
            legend_labels = models
        for (label, legend_label) in zip(models, legend_labels):
            for C in C_l:
                y_err_l = []
                y_l = []
                for delta in delta_l:
                    if delta == 0.:
                        exp = self.experiments_dict[label][K][C][attack_nodes][n_adv][delta_l[1]]
                        y, y_std = 1.0, 0
                    else:
                        exp = self.experiments_dict[label][K][C][attack_nodes][n_adv][delta]
                        y, y_std = exp.get_result("cert_node_frac")
                    y_l.append(y)
                    y_err_l.append(y_std)
                if xlogscale:
                    ax.set_xscale('log')
                    x = np.array(delta_l)
                else:
                    x = [i for i in range(len(delta_l))]
                
                label_str = r'{0}'.format(legend_label) #+ " " + str(C)
                ax.errorbar(x, y_l, yerr=y_err_l, marker="o", label=label_str, 
                            capsize=3, linewidth=1, markersize=4)
                self.set_xaxis_labels(ax, x, delta_l)
        ax.set_ylabel(r"Certified $\%$ of nodes", fontsize=20)
        ax.set_xlabel(r"$\delta$", fontsize=17, fontweight="bold")
        ax.yaxis.grid()
        ax.xaxis.grid()
        ax.legend()
        if savefig:
            CERTIFICATE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(CERTIFICATE_FIGURE_DIR/savefig, bbox_inches='tight', dpi=600)
        plt.show()
    
    def plot_network(self, dataset="karate_club", label="GCN", K=1.5, C=0.01, delta=0.2, seed=0):
        if dataset == "karate_club":
            _, A, y = get_karate_club()
            G = nx.from_numpy_array(A)
            exp = self.experiments_dict[label][K][C][delta]
            idx_labeled, idx_test, y_flip, y_is_robust = get_experiment(exp, seed)
            print('y label ', y[idx_labeled])
            print('y test ', y[idx_test])
            y_labeled = y[idx_labeled]
            y_test = y[idx_test]
            fig, ax = plt.subplots(1, 3, figsize=(12,4))
            pos = nx.fruchterman_reingold_layout(G, seed=10)
            nx.draw(G, ax=ax[0], node_color=y, with_labels=True, pos=pos)
            color_map = []
            for node in G:
                if node in idx_labeled:
                    color_map.append('blue')
                elif node in idx_test:
                    color_map.append('green')
            nx.draw(G, ax=ax[1], node_color=color_map, with_labels=True, pos=pos)
            color_map = []
            for node in G:
                if node in idx_labeled:
                    idx = idx_labeled.index(node)
                    if y_flip[idx] != y_labeled[idx]:
                        color_map.append('black')
                    else:
                        color_map.append('blue')
                elif node in idx_test:
                    idx = idx_test.index(node)
                    if y_is_robust[idx] == 1:
                        color_map.append('green')
                    else:
                        color_map.append('red')
            nx.draw(G, ax=ax[2], node_color=color_map, with_labels=True, pos=pos)

    def plot(self, name: str, attack: str, models: List[str], 
             errorbars: bool=True, ylabel: str=None, title: str=None,
             spacing: str="normal", legend_loc="best", legend_cols: int=None,
             budget: str=None, yspacing: str="normal", width=0.86, ratio=1.618,
             titlefont=20, fontweight="bold",
             K_l: List[float]=[0.1, 0.5, 1, 1.5, 2, 3, 4, 5]):
        """Plot relative or absolute over-robustness measure.

        Args:
            name (str): What measurement to plot:
                - over-robustness
                - relative-over-robustness
                - f_wrt_y (allows for BC model)
                - adversarial-robustness
                - relative-adversarial-robustness
                - validation-accuracy
                - test-accuracy
                - f1-robustness
                - f1-min-changes
            attack (str): 
            models (List[str]): White-list
            errorbars (bool): True or False
            ylabel: Label of y-axis. If not provided it is set to "name".
            title: Title of plot. If not provided it is set to "name".
            K_l:List[float]=[0.1, 0.5, 1, 1.5, 2, 5]. White-list
        """
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        #color_list = ['r', 
        #              'tab:green', 
        #              'b', 
        #              'lime', 
        #              'slategrey', 
        #              'k', 
        #              "lightsteelblue",
        #              "antiquewhite",
        #              ]
        #linestyle_list = ['-', '--', ':', '-.']
        #ax.set_prop_cycle(cycler('linestyle', linestyle_list)*
        #                  cycler('color', color_list))
        added_bayes = False
        for label, exp_by_k in self.model_iterator(attack, models):
            x = []
            y = []
            y_err = []
            y_bc = []
            y_err_bc = []
            for K, exp in exp_by_k.items():
                if K not in K_l:
                    continue
                x.append(K)
                value, std = exp.get_measurement(name, budget)
                y.append(value)
                y_err.append(std)
                if "BC" in models and not added_bayes:
                    if name == "f_wrt_y":
                        value, std = exp.get_measurement("g_wrt_y", budget)
                    elif name == "test-accuracy":
                        value, std = exp.get_measurement("test-accuracy-bayes")
                    else:
                        raise ValueError("BC requested but name not f_wrt_y")
                    y_bc.append(value)
                    y_err_bc.append(std)
            sort_ids = np.argsort(x)
            if spacing == "even":
                x = [i for i in range(len(K_l))]
            else:
                x = K_l
            y = np.array(y)[sort_ids]
            color, linestyle = self.get_style(label)
            if label == "GraphSAGE":
                label = "GraphSAGE"
            if label == "GraphSAGE+LP":
                label = "GraphSAGE+LP"
            if errorbars:
                y_err = np.array(y_err)[sort_ids]
                ax.errorbar(x, y, yerr=y_err, marker="o", color=color, linestyle=linestyle,
                            label=label, capsize=5, linewidth=2.5, markersize=8)
                if "BC" in models and not added_bayes:
                    y_bc = np.array(y_bc)[sort_ids]
                    y_err_bc = np.array(y_err_bc)[sort_ids] 
                    ax.errorbar(x, y_bc, yerr=y_err_bc, fmt="s:", label="Bayes Classifier", 
                    capsize=5, linewidth=2.5, markersize=8)
            else:
                ax.plot(x, y, marker="o",  color=color, linestyle=linestyle, 
                        label=label)
                if "BC" in models and not added_bayes:
                    y_bc = np.array(y_bc)[sort_ids]
                    ax.plot(x, y_bc, "s:", label="Bayes Classifier")
            added_bayes = True
        if ylabel is None:
            ylabel=name
        
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        if title is None:
            title=name
        if yspacing == "log":
            ax.set_yscale('log')
        if spacing == "log":
            ax.set_xscale('log')
            xticks = np.sort(K_l.append([0.2, 10]))
            ax.set_xticks(xticks, minor=True)
        elif spacing == "even":
            ax.xaxis.set_ticks(x, minor=False)
            xticks = [f"{K}" for K in K_l]
            ax.xaxis.set_ticklabels(xticks, fontsize=15, fontweight="bold")
            ax.set_xlim(left=-0.3)
        else:
            ax.set_xticks(K_l, minor=False)
            ax.set_xlim(left=0.)
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_xlabel("K", fontsize=17, fontweight="bold")
        ax.set_title(title, fontweight=fontweight, fontsize=titlefont)
        #ax.set_yticklabels([f"{round(i, 2):.2f}" for i in ax.get_yticks()], fontsize=13)
        ax.set_yticklabels([f"{round(i, 1):.1f}" for i in ax.get_yticks()], fontsize=15, fontweight="bold")
        #ax.set_xticklabels(ax.get_xticks(), fontsize=13)
        ax.yaxis.grid()
        ax.xaxis.grid()
        if legend_cols is None:
            ax.legend(loc=legend_loc)
        else:
            box = ax.get_position()
            #w = box.width * 0.8
            #h = w / 1.618
            #width = 1.618 * box.height
            ax.set_position([box.x0, box.y0, box.width*width, box.height])
            leg = ax.legend(loc=legend_loc, ncol=legend_cols, shadow=False,
                            bbox_to_anchor=(1, -0.23), frameon=False, markerscale=1,
                            prop=dict(size=12.3, weight="bold"))
            #leg.get_lines()[0].set_linestyle
            #for i in leg.legendHandles:
            #    i.set_linestyle(":")
            #ax.set_aspect(1.618)
        fig.set_tight_layout(True)
        plt.show()

    def starplot(self, name: str, attack: str, models: List[str], 
                 K: List[float], max_degree: int=None, logplot: bool=False,
                 errorbars: bool=True, ylabel: str=None, title: str=None,
                 bayes_label="Bayes"):
        """Generate starplot (w.r.t. degree plots).

        Args:
            name (str):
                - f_wrt_y
                - f_wrt_g
                - both
            attack (str): _description_
            models (List[str]): _description_
            K (List[float]): _description_
            errorbars (bool, optional): _description_. Defaults to True.
            ylabel (str, optional): _description_. Defaults to None.
            title (str, optional): _description_. Defaults to None.
        """
        if max_degree is None:
            max_degree = 0
            for label, K, exp in self.experiment_iterator(attack, models, [K]):  
                avg_f_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
                max_deg_ = max([int(deg) for deg in avg_f_wrt_y.keys()])
                if max_deg_ > max_degree:
                    max_degree = max_deg_
        #h, w = matplotlib.figure.figaspect(1.618 / 1.15)
        #fig, axs = plt.subplots(figsize=(w,h))
        fig, axs = plt.subplots()
        color_list = ['r', 'tab:green', 'b', 'lime', 'c', 'k', "antiquewhite"]
        linestyle_list = ['-', '--', ':', '-.']
        axs.set_prop_cycle(cycler('linestyle', linestyle_list)*
                           cycler('color', color_list))
        bayes_added = False
        for label, K, exp in self.experiment_iterator(attack, models, [K]):  
            avg_f_wrt_y = exp.robustness_statistics["avg_avg_gnn_robust_when_both"]
            std_f_wrt_y = exp.robustness_statistics["sem_avg_gnn_robust_when_both"]
            avg_g_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
            std_g_wrt_y = exp.robustness_statistics["sem_avg_bayes_robust_when_both"]
            avg_f_wrt_g = exp.robustness_statistics["avg_avg_gnn_wrt_bayes_robust"]
            std_f_wrt_g = exp.robustness_statistics["sem_avg_gnn_wrt_bayes_robust"]
            
            x = np.sort([int(i) for i in avg_f_wrt_y.keys()])
            x = x[x <= max_degree]
            ordered_avg_f_wrt_y = [avg_f_wrt_y[str(i)] for i in x]
            ordered_std_f_wrt_y = [std_f_wrt_y[str(i)] for i in x]
            ordered_avg_g_wrt_y = [avg_g_wrt_y[str(i)] for i in x]
            ordered_std_g_wrt_y = [std_g_wrt_y[str(i)] for i in x]
            ordered_avg_f_wrt_g = [avg_f_wrt_g[str(i)] for i in x]
            ordered_std_f_wrt_g = [std_f_wrt_g[str(i)] for i in x]
            color, linestyle = self.get_style(label)
            if errorbars:
                if not bayes_added:
                    axs.errorbar(x, ordered_avg_g_wrt_y, 
                                yerr=ordered_std_g_wrt_y, fmt="s:", label=f"{bayes_label}", capsize=3,
                                color="tab:olive")
                if name == "f_wrt_y" or name == "both":
                    axs.errorbar(x, ordered_avg_f_wrt_y,  
                                yerr=ordered_std_f_wrt_y, marker="o", label=f"{label}", capsize=3,
                                color=color, linestyle=linestyle)
                if name == "f_wrt_g" or name == "both":
                    axs.errorbar(ordered_avg_f_wrt_g, x, 
                                yerr=ordered_std_f_wrt_g, marker="o", label=f"{label} w.r.t. Bayes", capsize=3)
            else:
                if name == "f_wrt_y" or name == "both":
                    axs.plot(ordered_avg_f_wrt_y, x, marker='o', label=f"{label}")
                if not bayes_added:
                    axs.plot(ordered_avg_g_wrt_y, x, 's:', label=bayes_label)
                if name == "f_wrt_g" or name == "both":
                    axs.plot(ordered_avg_f_wrt_g, x, marker='o', label=f"{label} w.r.t. Bayes")
            bayes_added = True
        if ylabel is None:
            ylabel=name
        axs.set_ylabel(ylabel, fontsize=13)
        axs.set_xlabel("Node Degree", fontsize=13)
        if logplot:
            axs.set_yscale('log')
        if title is None:
            title=name
        axs.set_title(title, fontweight="bold", fontsize=15)
        start_x, end_x = axs.get_xlim()
        start_y, end_y = axs.get_ylim()
        # filling:
        for label, K, exp in self.experiment_iterator(attack, models, [K]):  
            avg_g_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
            x = np.sort([int(i) for i in avg_f_wrt_y.keys()])
            x = x[x <= max_degree]
            ordered_avg_g_wrt_y = [avg_g_wrt_y[str(i)] for i in x]
            start_y, end_y = axs.get_ylim()
            axs.fill_between(x, start_y, ordered_avg_g_wrt_y, interpolate=True, 
                                color='tab:olive', alpha=0.1)
            axs.fill_between(x, ordered_avg_g_wrt_y, end_y, 
                                interpolate=True, color='red', alpha=0.1)
            break

        #axs.xaxis.set_ticks(np.arange(0, end_x, step=1))
        axs.yaxis.set_ticks([1, 10, 100], ["1", "10", "100"])
        axs.set_xticklabels([int(i) for i in axs.get_xticks()], fontsize=10)
        axs.set_yticklabels(axs.get_yticks(), fontsize=10)
        #axs.xaxis.set_ticks_position('top')
        #axs.xaxis.set_label_position('top')
        box = axs.get_position()
        axs.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.9])
        axs.legend(loc="center left", ncol=1, shadow=False,
                    bbox_to_anchor=(1, 0.5), frameon=False, fontsize=12)
        axs.invert_yaxis()
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0)
        t = axs.text(
            4, 2, "Preserved\nSemantics", 
            rotation=0, size=14, color="tab:olive", fontstyle="oblique",
            fontweight="bold",
            bbox=bbox_props)
        t = axs.text(
            3, 38, "Changed\nSemantics", 
            rotation=0, size=14, color="r", fontstyle="oblique",
            fontweight="bold",
            bbox=bbox_props)
        #plt.grid(axis="both")
        plt.show()

