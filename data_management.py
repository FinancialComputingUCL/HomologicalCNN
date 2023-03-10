import time
import shutil

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import openml.datasets as open_data

import params
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

        self.numerical_features = None
        self.categorical_features = None

    def __get_feature_types(self, dataset):
        feature_types = dataset.features
        feature_names = feature_types.keys()
        numerical_features = []
        categorical_features = []
        for feature_name in feature_names:
            if feature_types[feature_name].data_type == 'numeric':
                numerical_features.append(feature_name)
            else:
                categorical_features.append(feature_name)

        return numerical_features, categorical_features

    def __download_open_ml_dataset(self):
        while True:
            try:
                try:
                    shutil.rmtree('/home/abriola/.cache/openml/org/openml/')
                except:
                    print('Unable to delete OpenML cache folder.')
                    pass
                dataset = open_data.get_dataset(self.dataset_id)
                self.X, self.y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
                break
            except:
                time.sleep(30)
                print(f'Downloading error for dataset {self.dataset_id}. Trying again in 30 secs...')
                continue

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

        # Isolate numerical features.
        '''self.X = self.X[self.numerical_features]
        self.X.reset_index(drop=True, inplace=True)
        self.X.columns = np.arange(0, self.X.shape[1]).tolist()'''

        # Isolate categorical features.
        '''self.X_cat = self.X[self.categorical_features]
        self.X_cat.reset_index(drop=True, inplace=True)
        self.X_cat.columns = np.arange(0, self.X_cat.shape[1]).tolist()'''

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
