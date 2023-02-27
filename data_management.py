import pandas as pd
import numpy as np
import networkx as nx

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import openml.datasets as open_data

import params
from tmfg_core import *
from tmfg_bootstrapped import *


class DataManager:
    def __init__(self, dataset_id, seed):
        self.dataset_id = dataset_id
        self.seed = seed
        np.random.seed(self.seed)

        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.__download_open_ml_dataset()
        self.__customize_data()

    def __download_open_ml_dataset(self):
        dataset = open_data.get_dataset(self.dataset_id)

        self.X, _, _, _ = dataset.get_data(dataset_format="dataframe")
        self.X.dropna(inplace=True)
        self.y = self.X.iloc[:, -1:]
        self.X.drop([self.y.columns[0]], axis=1, inplace=True)

        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(np.array(self.y).ravel())

        columns_names = []
        for n, i in enumerate(self.X.columns):
            columns_names.append(n)
        self.X.columns = columns_names

        if params.SHUFFLE_DATA_BEFORE_SPLITTING:
            self.X, self.y = shuffle(self.X, self.y)
            self.X.reset_index(drop=True, inplace=True)

    def __customize_data(self):
        upper_bound_train_test = int(len(self.X) * params.TEST_PERCENTAGE)
        upper_bound_train_val = int((len(self.X) - upper_bound_train_test) * params.VALIDATION_PERCENTAGE)

        self.X_train = self.X[:-upper_bound_train_test]
        self.y_train = self.y[:-upper_bound_train_test]

        self.X_val = self.X_train[-upper_bound_train_val:]
        self.y_val = self.y_train[-upper_bound_train_val:]

        self.X_train = self.X_train[:-upper_bound_train_val]
        self.y_train = self.y_train[:-upper_bound_train_val]

        self.X_test = self.X[-upper_bound_train_test:]
        self.y_test = self.y[-upper_bound_train_test:]

        self.X_train = pd.DataFrame(self.X_train)
        self.X_val = pd.DataFrame(self.X_val)
        self.X_test = pd.DataFrame(self.X_test)

    def get_data(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test


class HomologicalDataManager:
    def __init__(self, X_train, X_val, X_test, tmfg_iterations, tmfg_confidence, tmfg_similarity, seed):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.tmfg_iterations = tmfg_iterations
        self.tmfg_confidence = tmfg_confidence
        self.tmfg_similarity = tmfg_similarity

        self.seed = seed

        self.__get_stat_robust_tmfg()

    def __get_stat_robust_tmfg(self):
        cliques, separators, original_tmfg, _, adjacency_matrix = TMFG_Bootstrapped(df=self.X_train,
                                                                                    correlation_type=self.tmfg_similarity,
                                                                                    number_of_repetitions=self.tmfg_iterations,
                                                                                    confidence_level=self.tmfg_confidence,
                                                                                    parallel=True,
                                                                                    seed=self.seed).compute_tmfg_bootstrapping()

        c = nx.degree_centrality(adjacency_matrix)

        keys = np.array(list(c.keys()))
        values = np.array(list(c.values()))
        nodes_list = sorted(list(keys[values != 0]))

        self.X_train = self.X_train[nodes_list]
        self.X_val = self.X_val[nodes_list]
        self.X_test = self.X_test[nodes_list]
        self.number_selected_features = len(nodes_list)

    def get_homological_data(self):
        return self.X_train, self.X_val, self.X_test, self.number_selected_features
