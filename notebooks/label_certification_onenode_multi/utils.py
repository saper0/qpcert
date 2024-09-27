from collections.abc import Iterable
import copy
import math
from pathlib import Path
from pymongo import MongoClient
from typing import Any, Dict, Iterator, List, Tuple, Union

from cycler import cycler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

CERTIFICATE_FIGURE_DIR = Path('./figures/')

URI = "mongodb://sabanaya:bAvQwbOp@fs.kdd.in.tum.de:27017/sabanaya?authMechanism=SCRAM-SHA-1"


def assert_equal_dicts(d1: Dict[str, Any], d2: Dict[str, Any]):
    """Checks if every key of d1 is in d2 and they store same value."""
    for key in d1:
        assert key in d2
        assert d1[key] == d2[key]


def append_dict(source: Dict[str, Any], target: Dict[str, Any], 
                include_keys: List[str] = None, exclude_keys: List[str] = None):
    """Appends each element in source-dict to same key-element in target. 
    
    Assumes source-dict has as elements a single number, list, or a dict. 
    Possibility to ignore keys given in exclude_keys.
    """
    for key, item in source.items():
        if include_keys is not None:
            if key not in include_keys:
                continue
        if exclude_keys is not None:
            if key in exclude_keys:
                continue
        if isinstance(item, dict):
            if key not in target:
                target[key] = {}
            exclude_keys = ["idx_train", "idx_val", "idx_labeled", "idx_adv"]
            append_dict(item, target[key], None, exclude_keys)
        elif isinstance(item, list):
            if key not in target:
                target[key] = []
            if key == "y_pred_logit":
                target[key].append(item)
            else:
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


def average_dict(target: Dict[str, Any], exclude_keys: List[str] = None):
    """Recursively averages every value-holding element in target-dictionary. 
    
    Assumes target dictionary holds str-keys and either list or dict items. 
    Every list-item is averaged. If element is a dict, assumes it is 
    shallow only containing lists.
    """
    keys = [key for key in target]
    for key in keys:
        if exclude_keys is not None:
            if key in exclude_keys:
                continue
        item = target[key]
        if isinstance(item, dict):
            avg_dict, std_dict, sem_dict = average_subdict(item)
            target["sem_" + key] = sem_dict
            target["avg_" + key] = avg_dict
            target["std_" + key] = std_dict
        else:
            if len(target[key]) == 0:
                print(len(target[key]))
                print(target[key])
                print(key)
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


def prepare_experiment_dict(exp_dict):
    """Prepare experiment dict for later adding of results.
    
    Changes all scalar results to list.
    """
    exp_dict["result"]["y_pred"] = np.argmax(exp_dict["result"]["y_pred_logit"])
    for key in exp_dict["result"]:
        if not isinstance(exp_dict["result"][key], list) or key == "y_pred_logit":
            val = exp_dict["result"][key]
            exp_dict["result"][key] = [val]


class Experiment:
    """An experiment refers to the (robustness) results optained by a 
    particular model on K averaged over multiple seeds."""
    def __init__(self, experiment: Dict[str,Any]):
        prepare_experiment_dict(experiment)
        self.individual_experiments = [experiment]
        self.id = experiment["_id"]
        print(self.id)
        self.hyperparameters = experiment["config"]
        self.dataset = self.hyperparameters["data_params"]["dataset"]
        self.label = self.hyperparameters["model_params"]["label"]
        self.C = float(self.hyperparameters["model_params"]["regularizer"])
        params = self.hyperparameters["certificate_params"]
        self.delta = float(params["delta"])

    def add_experiment(self, exp: Dict[str, Any]):
        """Add experiment dict to Experiment object."""
        n_seed_found = False
        for experiment in self.individual_experiments:
            if experiment["config"]["seed"] == exp["config"]["seed"]:
                Experiment.assert_same_hyperparameters([experiment, exp])
                exp["result"]["y_pred"] = np.argmax(exp["result"]["y_pred_logit"])
                append_dict(exp, experiment, ["result"])
                n_seed_found = True
        if not n_seed_found:
            Experiment.assert_same_hyperparameters([self.individual_experiments[0], 
                                                    exp])
            prepare_experiment_dict(exp)
            self.individual_experiments.append(exp)
        self.average_result_statistics()

    @staticmethod
    def assert_same_hyperparameters(
        individual_experiments: List[Dict[str, Any]]
    ) -> None:
        """Sanity check if all given experiments indeed have the same 
        configuration."""
        if len(individual_experiments) == 1:
            return
        data_params_l = []
        model_params_l = []
        certificate_params_l = []
        for experiment in individual_experiments:
            data_params = experiment["config"]["data_params"]
            if "seed" in data_params["specification"]:
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
            result = experiment["result"]
            exclude_keys = ["idx_train", "idx_val", "idx_labeled", "idx_adv", "y_flip"]
            append_dict(result, self.results, None, exclude_keys)
        average_dict(self.results, ["y_pred_logit"])

    def get_result(self, key: str) -> Tuple[float, float]:
        """Return average & std of metric 'key' over all seeds."""
        return np.mean(self.results[key]).item(), np.std(self.results[key]).item()
    
    def get_test_accuracy(self) -> Tuple[float, float]:
        """Test accuracy over target nodes across seeds."""
        n_acc_l = []
        for experiment in self.individual_experiments:
            n_corr = 0
            accuracy_test_l = experiment["result"]["accuracy_test"]
            n_acc_l.append( sum(accuracy_test_l) / len(accuracy_test_l))
        return np.mean(n_acc_l).item(), np.std(n_acc_l).item()

    def get_robust_accuracy(self) -> Tuple[float, float]:
        """
        For certificates returns certified accuracy.
        For attacks returns roubst accuracy.
        """
        n_robust_acc_l = []
        for experiment in self.individual_experiments:
            n_robust_acc = 0
            result = experiment["result"]
            n_test = len(result["accuracy_test"])
            for y_true, y_pred, y_robust in zip(result["y_true_cls"],
                                                result["y_pred"],
                                                result["y_is_robust"]):
                if y_pred == y_true and y_robust:
                    n_robust_acc += 1
            n_robust_acc_l.append(n_robust_acc / n_test)
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

    def load_experiment_dict(self, id: int, collection: str) -> Dict[str, Any]:
        """Return result-dict of experiment with ID id."""
        return self.db[collection].find_one({'_id': id})

    def load_experiments(
            self, start_id: int, end_id: int, label_filter: str=None, 
            collection: str="runs"
        ) -> Dict:
        """Return Experiments between start_id and end_id with given label.
        
        Assumes that one experiment consists of multiple seeds which are stored
        consecutively in the mongodb in a given collection.
        """
        for idx in range(start_id, end_id+1):
            if idx % 1000 == 0:
                print(f"Loading {idx}th experiment...")
            exp_dict = self.load_experiment_dict(idx, collection)
            if exp_dict is None:
                continue
            if exp_dict["status"] != "COMPLETED":
                continue
            hyperparameters = exp_dict["config"]
            label = hyperparameters["model_params"]["label"]
            if label_filter is not None:
                if label != label_filter:
                    continue
            C = float(hyperparameters["model_params"]["regularizer"])
            delta = float(hyperparameters["certificate_params"]["delta"])
            if label not in self.experiments_dict:
                self.experiments_dict[label] = {}
            if C not in self.experiments_dict[label]:
                self.experiments_dict[label][C] = {}
            if delta not in self.experiments_dict[label][C]:
                exp = Experiment(exp_dict)
                self.experiments_dict[label][C][delta] = exp
            else:
                self.experiments_dict[label][C][delta].add_experiment(exp_dict)

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
                self.load_experiments(start_id, end_id, label, 
                                      exp_spec["collection"])

    def get_style(self, label: str):
        color_dict = {
            "APPNP_alpha1": 'slategrey', #MLP
            "MLP": 'slategrey', #MLP
            "GCN": 'tab:green', 
            "GCN_sym": 'tab:green', 
            "APPNP_alpha0": "plum",
            "APPNP_alpha0.1": "tab:brown",
            "APPNP_0.1": "tab:brown",
            "APPNP_alpha0.1_row": "tab:brown",
            "APPNP_alpha0.2": "r",
            "APPNP_0.2": "r",
            "APPNP": 'r', #lime 
            "APPNP_alpha0.3": "tab:olive",
            "APPNP_0.3": "tab:olive",
            "APPNP_alpha0.3_row": "tab:olive",
            "APPNP_alpha0.5": "darkslategrey",
            "APPNP_0.5": "darkslategrey",
            "SGC": "blue",
            "SGC_sym": "blue",
            "GCN_skippc": "lime", #k
            "GCN_skippc_linear": "lime", #k
            "GCN_skippc_relu+2": "lime",
            "GCN_skipalpha": "plum", #"wheat",
            "GCN_skipalpha_linear_alpha0.2": "wheat",
            "GCN_skipalpha_linear_alpha0.1": "wheat",
            "GCN_skipalpha_relu_alpha0.2+2": "wheat",
            "GCN_skipalpha_linear_alpha0.1": "steelblue",
            "GCN_skipalpha_relu_alpha0.1+2": "steelblue",
            "GIN": "darkslateblue",
            "GraphSAGE": "darkred",
            # "GAT": "slategrey",
            # "GATv2": "k",
            # "GraphSAGE": "lightsteelblue",
            # "LP": "wheat",
        }
        linestyle_dict = {
            "LP": '--',
            "SGC_sym": ":",
            "GCN_sym": ":",
            "APPNP_alpha0.1_row": "dashed",
            "APPNP_alpha0.3_row": "dashed",
            "MLP": 'dashed'
        }
        use_color=""
        linestyle="-"
        for key, color in color_dict.items():
            sep_labels = key.split("+")
            if sep_labels[0] == label:
                use_color = color
                if len(sep_labels) == 2 or sep_labels[0] == "LP":
                    linestyle = "--"
        for key, linestyle_ in linestyle_dict.items():
            if label == key:
                linestyle = linestyle_
        return use_color, linestyle
    
    def set_color_cycler(self, ax):
        color_list = ['r', 
                      'tab:green', 
                      'b', 
                      'lime', 
                      'slategrey', 
                      'k', 
                      "lightsteelblue",
                      "antiquewhite",
                      ]
        linestyle_list = ['-', '--', ':', '-.']
        ax.set_prop_cycle(cycler('linestyle', linestyle_list)*
                          cycler('color', color_list))

    def set_xaxis_labels(self, ax, x_ticks, x_labels, fontsize=12):
        ax.xaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_ticks(x_ticks, minor=False)
        xticks = [f"{label}" for label in x_labels]
        ax.xaxis.set_ticklabels(xticks, fontsize=fontsize, fontweight="bold")
        ax.set_xlim(left=-0.3)
    
    def set_xaxis_labels_logscale(self, ax, x_ticks, x_labels, fontsize):
        ax.xaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_ticks(x_ticks, minor=False)
        xticks = [f"{label}" for label in x_labels]
        ax.xaxis.set_ticklabels(xticks, fontsize=fontsize)

    def plot_robust_acc_delta(self, models: List[str], C_l: List[float], 
                              delta_l: List[float],
                              width=1, ratio=1.618, spacing="even"):
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        self.set_color_cycler(ax)
        print([key for key in self.experiments_dict])
        for label in models:
            for C in C_l:
                if C not in self.experiments_dict[label]:
                    continue
                y_err_l = []
                y_l = []
                for delta in delta_l:
                    if delta == 0.:
                        exp = self.experiments_dict[label][C][delta_l[1]]
                        y, y_std = exp.get_test_accuracy()
                    else:
                        exp = self.experiments_dict[label][C][delta]
                        y, y_std = exp.get_robust_accuracy()
                    y_l.append(y)
                    y_err_l.append(y_std)
                if spacing == "even":
                    x = [i for i in range(len(delta_l))]
                elif spacing == "log":
                    ax.set_xscale('log')
                    x = delta_l
                label_str = label + " " + str(C)
                ax.errorbar(x, y_l, yerr=y_err_l, marker="o", label=label_str, 
                            capsize=3, linewidth=1, markersize=4)
                self.set_xaxis_labels(ax, x, delta_l)
        ax.set_ylabel("Certified Accuracy", fontsize=20)
        ax.set_xlabel(r"$\delta$", fontsize=17, fontweight="bold")
        ax.yaxis.grid()
        ax.xaxis.grid()
        ax.legend()
        plt.show()

    def plot_robust_acc_delta_v2(self, models: List[str], C_l: List[float], 
                              delta_l: List[float],
                              legend_labels: List[str]=[],
                              width=1, ratio=1.618, 
                              xlogscale: bool=False,
                              ylogscale: bool=False,
                              savefig: str=None,
                              savedir: Path=None,
                              label_fontsize=16,
                              legend_fontsize=12,
                              ticks_fontsize=10,
                              markersize=4,
                              capsize=3,
                              linewidth=1,
                              framealpha=1.0):
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        # self.set_color_cycler(ax)
        if len(legend_labels) != len(models):
            legend_labels = models
        C_l_None_Flag = C_l
        for (label, legend_label) in zip(models, legend_labels):
            if C_l_None_Flag == None:
                C_l = [key for key in self.experiments_dict[label]]
            for C in C_l:
                y_err_l = []
                y_l = []
                for delta in delta_l:
                    if delta == 0.:
                        exp = self.experiments_dict[label][C][delta_l[1]]
                        y, y_std = exp.get_test_accuracy()
                    else:
                        exp = self.experiments_dict[label][C][delta]
                        y, y_std = exp.get_robust_accuracy()
                    y_l.append(y)
                    y_err_l.append(y_std)
                if ylogscale:
                    ax.set_yscale('log')
                if xlogscale:
                    ax.set_xscale('log')
                    x = np.array(delta_l)
                    if x[0] == 0:
                        x[0] = 0.005
                    self.set_xaxis_labels_logscale(ax, x, delta_l, ticks_fontsize)
                else:
                    x = [i for i in range(len(delta_l))]
                    self.set_xaxis_labels(ax, x, delta_l, ticks_fontsize)
                
                label_str = r'{0}'.format(legend_label) #+ " " + str(C)
                color, linestyle = self.get_style(label)
                ax.errorbar(x, y_l, yerr=y_err_l, marker="o", label=label_str, 
                            color=color, linestyle=linestyle,
                            capsize=capsize, linewidth=linewidth, 
                            markersize=markersize)
        ax.set_ylabel("Certified Accuracy", fontsize=label_fontsize)
        ax.set_xlabel(r"Perturbation budget $\delta$", fontsize=label_fontsize)
        ax.yaxis.grid()
        ax.xaxis.grid()
        ax.legend(fontsize=legend_fontsize, framealpha=framealpha)
        ax.tick_params(labelsize=ticks_fontsize)
        if savefig:
            if savedir is None:
                savedir = CERTIFICATE_FIGURE_DIR
            savedir.mkdir(parents=True, exist_ok=True)
            plt.savefig(savedir/savefig, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def plot_robust_auc(self, models: List[str], C_l: List[float], 
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
        C_l_None_Flag = C_l
        aoc_l = []
        aoc_lb = []
        aoc_ub = []
        for (label, legend_label) in zip(models, legend_labels):
            if C_l_None_Flag == None:
                C_l = [key for key in self.experiments_dict[label]]
            for C in C_l:
                y_err_l = []
                y_l = []
                for delta in delta_l:
                    if delta == 0.:
                        exp = self.experiments_dict[label][C][delta_l[1]]
                        y, y_std = exp.get_test_accuracy()
                    else:
                        exp = self.experiments_dict[label][C][delta]
                        y, y_std = exp.get_robust_accuracy()
                    y_l.append(y)
                    y_err_l.append(y_std)
                x = np.array(delta_l)
                aoc_l.append(np.trapz(y_l, x))
                aoc_lb.append(np.trapz(np.array(y_l) - np.array(y_err_l), x))
                aoc_ub.append(np.trapz(np.array(y_l) + np.array(y_err_l), x))
        # scale by most robust model
        if scale:
            aoc_max = np.max(aoc_l)
            aoc_l = np.array(aoc_l) / np.max(aoc_l)
            idx_sorted = np.argsort(aoc_l)[::-1]
            aoc_lb = np.array(aoc_lb) / aoc_max
            aoc_ub = np.array(aoc_ub) / aoc_max
            x = np.arange(len(aoc_l))
            ax.set_xticks(x)
            ax.set_xticklabels(legend_labels)
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

    def print_clean_acc(self, models: List[str], C_l: List[float], delta_l: List[float]):
        for label in models:
            C_l = [key for key in self.experiments_dict[label]]
            for C in C_l:
                for delta in delta_l:
                    if delta == 0.:
                        exp = self.experiments_dict[label][C][delta_l[1]]
                    else:
                        exp = self.experiments_dict[label][C][delta]
                    acc_mean, acc_std = exp.get_test_accuracy()
                    print(f"{label} delta={delta} Target Acc. {acc_mean:.3f}+-{acc_std:.3f}")
    
    def print_cert_ratio(self, models: List[str], C_l: List[float], attack_nodes: str, 
                         n_adv: int, delta: float):
        for label in models:
            for C in C_l:
                exp = self.experiments_dict[label][C][delta]
                n_robust_acc_l = []
                for experiment in exp.individual_experiments:
                    n_robust_acc = 0
                    result = experiment["result"]
                    n_test = len(result["accuracy_test"])
                    print(n_test, len(result["y_is_robust"]))
                    n_robust_acc_l.append(sum(result["y_is_robust"]) / n_test)
                cert_ratio = np.mean(n_robust_acc_l).item()
                cert_ratio_std = np.std(n_robust_acc_l).item()
                print(f"{label} C={C:.5f} Cert. Ratio {cert_ratio:.3f}+-{cert_ratio_std:.3f}")
    
    def get_test_acc(self, models: List[str], C_l: List[float], attack_nodes: str, 
                     n_adv: int, delta: float):
        for label in models:
            for C in C_l:
                exp = self.experiments_dict[label][C][delta]
                test_acc_l = []
                for experiment in exp.individual_experiments:
                    test_acc = 0
                    result = experiment["result"]
                    n_test = len(result["accuracy_test"])
                    for y_true, y_pred, y_robust in zip(result["y_true_cls"],
                                                        result["y_pred"],
                                                        result["y_is_robust"]):
                        #print(y_pred_logit)
                        #print(np.argmax(y_pred_logit))
                        #assert False
                        if y_pred == y_true:
                            test_acc += 1
                    test_acc_l.append(test_acc / n_test)
                acc = np.mean(test_acc_l).item()
                acc_std = np.std(test_acc_l).item()
                print(f"{label} C={C:.5f} Cert. Ratio {acc:.3f}+-{acc_std:.3f}")
    