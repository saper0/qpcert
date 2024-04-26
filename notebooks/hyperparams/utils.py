from pymongo import MongoClient
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


URI = "mongodb://sabanaya:bAvQwbOp@fs.kdd.in.tum.de:27017/sabanaya?authMechanism=SCRAM-SHA-1"


def assert_equal_dicts(d1: Dict[str, Any], d2: Dict[str, Any]):
    """Checks if every key of d1 is in d2 and they store same value."""
    for key in d1:
        assert key in d2
        assert d1[key] == d2[key]
        

class Experiment:
    """An experiment refers to the results optained by a particular 
    hyperparameter setting averaged over multiple seeds.
    
    Allows comparison of experiments by average validation loss. """
    def __init__(self, experiment_list: List[Dict[str, Any]]):
        assert len(experiment_list) >= 0
        self.individual_experiments = experiment_list
        self.id = experiment_list[0]["_id"]
        self.hyperparameters = experiment_list[0]["config"]
        self.assert_same_hyperparameters(self.individual_experiments)
        self.extract_results(self.individual_experiments)

    def assert_same_hyperparameters(self, experiment_list: List[Dict[str, Any]]) -> None:
        """Sanity check if all experiments summarized by an instance of this 
        class indeed have the same configuration."""
        data_params_l = []
        spec_params_l = []
        model_params_l = []
        for experiment in experiment_list:
            data_params = experiment["config"]["data_params"]
            spec_params = data_params["specification"]
            del spec_params["seed"]
            data_params_l.append(data_params)
            spec_params_l.append(spec_params)
            model_params_l.append(experiment["config"]["model_params"])
        for i in range(1, len(experiment_list)):
            assert_equal_dicts(spec_params_l[0], spec_params_l[i])
            assert_equal_dicts(data_params_l[0], data_params_l[i])
            assert_equal_dicts(model_params_l[0], model_params_l[i])

    def extract_results(self, experiment_list: List[Dict[str, Any]]):
        """Collect important training results into a list."""
        self.val_acc_mean_l = []
        self.val_acc_std_l = []
        self.val_acc_all_l = []
        self.trn_acc_l = []
        self.test_acc_l = []
        for experiment in experiment_list:
            results = experiment["result"]
            val_acc_l = np.array(results["val_acc_l"])
            for val_acc in results["val_acc_l"]:
                self.val_acc_all_l.append(results["val_acc_l"])
            self.val_acc_mean_l.append(val_acc_l.mean())
            self.val_acc_std_l.append(val_acc_l.std())
            self.trn_acc_l.append(results["trn_acc"])
            self.test_acc_l.append(results["test_acc"])

    def to_dict(self, hyperparams_list: List[str]) -> Dict[str, Any]:
        """Genereates a dict representation of the experiment.
        
        Stores in "_id" the ID of the experiment. other keys are the results 
        and selected hyperparameter settings.
        """
        dict_repr = {}
        dict_repr["_id"] = self.id
        dict_repr["ValAcc"] = f"{self.validation_accuracy:.4f} " \
                              f"+- {self.validation_accuracy_std:.4f}"
        dict_repr["TrnAcc"] = f"{self.training_accuracy:.4f} " \
                              f"+- {self.training_accuracy_std:.4f}"
        for param in hyperparams_list:
            for config in [self.hyperparameters["data_params"],
                           self.hyperparameters["model_params"]]:
                if param in config:
                    dict_repr[param] = config[param]
        return dict_repr

    @staticmethod
    def to_dataframe(experiment_list: List["Experiment"], 
                     hyperparams_list: List[str]):
        experiment_dicts = []
        index = []
        for experiment in experiment_list:
            experiment_dict = experiment.to_dict(hyperparams_list)
            index.append(experiment_dict["_id"])
            del experiment_dict["_id"]
            experiment_dicts.append(experiment_dict)
        return pd.DataFrame(experiment_dicts, index=index)

    @property
    def validation_accuracy(self):
        return float(np.mean(self.val_acc_mean_l))

    @property
    def validation_accuracy_std(self):
        return float(np.std(self.val_acc_mean_l))

    @property
    def training_accuracy(self):
        return float(np.mean(self.trn_acc_l))

    @property
    def training_accuracy_std(self):
        return float(np.std(self.trn_acc_l))

    @property
    def test_accuracy(self):
        return float(np.mean(self.test_acc_l))

    @property
    def test_accuracy_std(self):
        return float(np.std(self.test_acc_l))
    
    @property
    def dataset(self):
        return self.hyperparameters["data_params"]["dataset"]
    
    @property
    def pred_method(self):
        return self.hyperparameters["model_params"]["pred_method"]
            
    # The six rich comparison methods
    def __lt__(self, other):
        return self.validation_accuracy < other.validation_accuracy
    
    def __le__(self, other):
        return self.validation_accuracy <= other.validation_accuracy
    
    def __eq__(self, other):
        return self.validation_accuracy == other.validation_accuracy
    
    def __ne__(self, other):
        return self.validation_accuracy != other.validation_accuracy
    
    def __gt__(self, other):
        return self.validation_accuracy > other.validation_accuracy
    
    def __ge__(self, other):
        return self.validation_accuracy >= other.validation_accuracy

    def __str__(self):
        print_str = "ID: " + str(self.individual_experiments[0]["_id"])
        print_str += f" ValAcc: {self.validation_accuracy*100:.2f} +- {self.validation_accuracy_std*100:.2f}"
        print_str += f" TrnAcc: {self.training_accuracy*100:.2f} +- {self.training_accuracy_std*100:.2f}"
        print_str += f" TestAcc: {self.test_accuracy*100:.2f} +- {self.test_accuracy_std*100:.2f}"
        return print_str


class ExperimentLoader:
    """Manages a mongodb-collection-connection and access to its data.
    
    Assumes same experiments with different seeds are stored consecutively.
    """
    def __init__(self, collection="runs", uri=URI):
        """Establish connection to a given mongodb collection."""
        self.client = MongoClient(uri)
        self.db = self.client.sabanaya
        self.collection = self.db[collection]

    def load_experiment_dict(self, id: int) -> Dict[str, Any]:
        """Return result-dict of experiment with ID id."""
        return self.collection.find_one({'_id': id})

    def load_experiment(self, start_id: int, n_seeds: int, use_seeds) -> Experiment:
        """Return experiment with ID id."""
        exp_dict_l = []
        for i in range(n_seeds):
            if use_seeds is not None and i >= use_seeds:
                continue
            exp_dict_l.append(self.load_experiment_dict(start_id + i))
        return Experiment(exp_dict_l)

    def load_experiments(self, start_id: int, end_id: int, n_seeds: int, 
                         pred_method: str, dataset: str, use_seeds: int) -> List[Experiment]:
        """Return Experiments between start_id and end_id on dataset.
        
        Assumes that one experiment consists of multiple seeds which are stored
        consecutively in the mongodb. 
        """
        experiment_ids = [i for i in range(start_id, end_id + 1, n_seeds)]
        experiments = [self.load_experiment(i, n_seeds, use_seeds) for i in experiment_ids]
        experiments_on_dataset = []
        for experiment in experiments:
            #if experiment.hyperparameters["model_params"]["hidden_channels"] != 64:
            #    continue
            if experiment.dataset == dataset and experiment.pred_method == pred_method:
                experiments_on_dataset.append(experiment)
        experiments_on_dataset.sort(reverse=True)
        return experiments_on_dataset


def get_best_hyperparams(
    start_id: int, end_id: int, n_seeds: int, hyperparams: List[str], 
    pred_method: str, dataset: str, collection: str="runs", use_seeds: int=None,
) -> pd.DataFrame:
    """Return experiments ordered by performance for a particular K or if K is
    a list, return a list of the best experiments for each K.

    Args:
        start_id (int): ID of first experiment in hyperparameter search.
        end_id (int): ID of last experiment in hyperparameter search.
        n_seeds (int): Number of different seeds used for each hpyerparameter 
            setting.
        hyperparams (List[str]): Hyperparameters to dislay in dataframe.
        dataset: str
        collections (str): Default: "runs". MongoDB-collection of Experiments
        use_seeds: If given, of the first use_seeds of n_seeds will be used.
    Returns:
        pd.DataFrame: _description_
    """
    exp_loader = ExperimentLoader(collection=collection)
    experiments = exp_loader.load_experiments(start_id, end_id, n_seeds, 
                                              pred_method, dataset, use_seeds)
    print("---- Details of best experiment ----")
    print(experiments[0])
    print("------------------------------------")
    return Experiment.to_dataframe(experiments, hyperparams)
