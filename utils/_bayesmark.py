import os
import json
import torch
import random
import pickle
import numpy as np
import pandas as pd
from bayesmark.bbox_utils import get_bayesmark_func
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from ._domain import UniformDomain

class BayesmarkExpRunner:
    def __init__(self, task_context, dataset, seed):
        self.seed = seed
        self.model = task_context['model']
        self.task = task_context['task']
        self.metric = task_context['metric']
        self.dataset = dataset
        self.hyperparameter_constraints = task_context['hyperparameter_constraints']
        self.bbox_func = get_bayesmark_func(self.model, self.task, dataset['test_y'])
        
    def evaluate_point(self, candidate_config):
        np.random.seed(self.seed)
        random.seed(self.seed)

        X_train, X_test, y_train, y_test = self.dataset['train_x'], self.dataset['test_x'], self.dataset['train_y'], self.dataset['test_y']

        for hyperparam, value in candidate_config.items():
            if self.hyperparameter_constraints[hyperparam][0] == 'int':
                candidate_config[hyperparam] = int(value)

        if self.task == 'regression':
            mean_ = np.mean(y_train)
            std_ = np.std(y_train)
            y_train = (y_train - mean_) / std_
            y_test = (y_test - mean_) / std_

        model = self.bbox_func(**candidate_config)
        scorer = get_scorer(self.metric)

        model = self.bbox_func(**candidate_config)  
        model.fit(X_train, y_train)
        generalization_score = scorer(model, X_test, y_test)

        return generalization_score
    
class BayesMarkFunc:
    def __init__(self, model, dataset, seed, dtype = torch.float64):
        data = self.read_dataset(dataset)
        task_context = self.read_json(model)
        self.benchmark = BayesmarkExpRunner(task_context, data, seed)
        self.dtype = dtype
        
    def read_dataset(self, dataset):
        self.dataset = dataset
        pickle_fpath = f'bayesmark/data/{dataset}.pickle'
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)
        return data
        
    def read_json(self, model):
        json_fpath =f"bayesmark/configs/bayesmark.json"
        with open(json_fpath) as f:
            read = json.load(f)
            
        if self.dataset == "diabetes":
            task = "regression"
            metric = "neg_mean_squared_error"
            self.n_categorical = 0
            self.n_continuous = 10
            self.n_targets = 346 - 25 + 1
            self.class_dist = "integer from 25 to 346"
            self.global_minimum = 0.536
        elif self.dataset == "digits":
            task = "classification"
            metric = "accuracy"
            self.n_categorical = 64
            self.n_continuous = 0
            self.n_targets = 16 - 0+ 1
            self.class_dist = "integer from 0 to 16"
            self.global_minimum = -1
        elif self.dataset == "Griewank":
            task = "regression"
            metric = "neg_mean_squared_error"
            self.n_categorical = 0
            self.n_continuous = 15
            self.n_targets = "continuous"
            self.class_dist = "float"
            self.global_minimum = 0.55
        elif self.dataset == "KTablet":
            task = "regression"
            metric = "neg_mean_squared_error"
            self.n_categorical = 0
            self.n_continuous = 15
            self.n_targets = "continuous"
            self.class_dist = "float"
            self.global_minimum = 0.6
        
        task_context = {'model': model, 'task': task, 'metric': metric}
        task_context['hyperparameter_constraints'] = read[model]
        return task_context
    
    def transform(self, x, value):
        if (value[0] == 'float') and (value[1] == 'log'):
            return torch.tensor(x).to(self.dtype).log10()
        elif (value[1] == 'linear'):
            return torch.tensor(x).to(self.dtype)
        elif (value[1] == 'logit'):
            X = torch.tensor(x).to(self.dtype)
            return (X / (1 - X)).log()
        else:
            raise ValueError
            
    def inv_transform(self, X, value):
        if value[1] == 'log':
            return (10**X).item()
        elif (value[0] == 'float') and (value[1] == 'linear'):
            return X.item()
        elif (value[0] == 'int') and (value[1] == 'linear'):
            return X.round().int().item()
        elif (value[1] == 'logit'):
            return (X.exp() / (1 + X.exp())).item()
        else:
            raise ValueError
            
    def type_transform(self, X, value):
        if value[0] == 'float':
            return float(X)
        elif value[0] == 'int':
            return int(X)
        else:
            raise ValueError
    
    def setup_domain(self):
        bounds = torch.vstack([
            self.transform(value[-1], value) for value in self.benchmark.hyperparameter_constraints.values()
        ]).T
        domain = UniformDomain(bounds)
        return domain
    
    def convert_to_config(self, X, transform=True):
        d = zip(
            self.benchmark.hyperparameter_constraints.keys(),
            self.benchmark.hyperparameter_constraints.values(),
        )
        if transform:
            config = {
                key : self.inv_transform(X[idx], value)
                for idx, (key, value) in enumerate(d)
            }
        else:
            config = {
                key : self.type_transform(X[idx], value)
                for idx, (key, value) in enumerate(d)
            }
        return config
    
    def evaluate(self, X, transform=True):
        #if len(X.shape) == 2:
        #    if not X.shape[0] == 1:
        #        raise ValueError("single evaluation")
        
        candidate_config = self.convert_to_config(X, transform=transform)
        y = self.benchmark.evaluate_point(candidate_config)
        return -1 * y
    
    def __call__(self, X, transform=True):
        if len(X.shape) == 2:
            if not X.shape[0] == 1:
                return torch.tensor([self.evaluate(x, transform=transform) for x in X]).to(self.dtype)
            else:
                return torch.tensor([self.evaluate(X.squeeze(), transform=transform)]).to(self.dtype)
        else:
            return torch.tensor([self.evaluate(X, transform=transform)]).to(self.dtype)
