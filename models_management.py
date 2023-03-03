import math

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils import shuffle

from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from hyperopt import tpe, hp, Trials
from hyperopt.fmin import fmin

import params
from data_management import *
from models_utils import *
from HCNN import *


class ModelsManager:
    def __init__(self, model, seed, dataset_id, X_train, X_val, X_test, y_train, y_val, y_test):
        self.model = model
        self.seed = seed
        self.dataset_id = dataset_id

        ### === Fix seed === ###
        np.random.seed(self.seed)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.available_models = ['RandomForest',
                                 'XGBoost',
                                 'CatBoost',
                                 'LightGBM',
                                 'TabularTransformer',
                                 'TabNet',
                                 'HCNN',
                                 'HRandomForest',
                                 'HXGBoost',
                                 'HCatBoost',
                                 'HLightGBM',
                                 'HTabularTransformer',
                                 'HTabNet']
        assert self.model in self.available_models, 'Model not recognised'

        self.post_opt_X_train = None
        self.post_opt_y_train = None

        self.tmfg_similarities_options = ['pearson', 'spearman']
        self.tmfg_pvalues_options = [5, 25, 50, 75, 95, 99]

    # === Random Forest Optimization === #

    def random_forest_objective(self, optimization_parameters):
        n_estimators = int(optimization_parameters['n_estimators'])
        max_depth = int(optimization_parameters['max_depth'])
        min_samples_leaf = int(optimization_parameters['min_samples_leaf'])
        min_samples_split = int(optimization_parameters['min_samples_split'])

        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf,
                                       min_samples_split=min_samples_split)

        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        try:
            model.fit(scaled_X_train, self.y_train)
            preds = model.predict(scaled_X_val)
            score = f1_score(self.y_val, preds, average='macro')
            return -score
        except:
            return 0

    @measure_execution_time
    def random_forest_optimize(self, trial):
        optimization_parameters = {'n_estimators': hp.quniform('n_estimators', 100, 4000, 200),
                                   'max_depth': hp.quniform('max_depth', 10, 50, 10),
                                   'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 10, 2),
                                   'min_samples_split': hp.quniform('min_samples_split', 2, 10, 2)}
        best = fmin(fn=self.random_forest_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def random_forest_out_of_sample_test(self, best_hopt):
        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = RandomForestClassifier(n_estimators=int(best_hopt['n_estimators']),
                                       max_depth=int(best_hopt['max_depth']),
                                       min_samples_leaf=int(best_hopt['min_samples_leaf']),
                                       min_samples_split=int(best_hopt['min_samples_split']))

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs

    def random_forest_manager(self):
        root_folder = f'./Homological_FS/RandomForestClassifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.random_forest_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')

        preds, probs = self.random_forest_out_of_sample_test(best_hopt=best_hopt)
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === XGBoost Optimization === #

    def xgboost_objective(self, optimization_parameters):
        learning_rate = optimization_parameters['learning_rate']
        max_dept = int(optimization_parameters['max_depth'])
        n_estimators = int(optimization_parameters['n_estimators'])
        subsample = optimization_parameters['subsample']
        colsample_bytree = optimization_parameters['colsample_bytree']
        colsample_bylevel = optimization_parameters['colsample_bylevel']
        gamma = optimization_parameters['gamma']
        alpha = optimization_parameters['alpha']
        lambda_ = optimization_parameters['lambda']

        model = XGBClassifier(n_estimators=n_estimators,
                              max_depth=max_dept,
                              learning_rate=learning_rate,
                              subsample=subsample, colsample_bytree=colsample_bytree,
                              colsample_bylevel=colsample_bylevel,
                              gamma=gamma, alpha=alpha, lambda_=lambda_,
                              n_jobs=8)

        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        try:
            model.fit(scaled_X_train, self.y_train)
            preds = model.predict(scaled_X_val)
            score = f1_score(self.y_val, preds, average='macro')
            return -score
        except:
            return 0

    @measure_execution_time
    def xgboost_optimize(self, trial):
        optimization_parameters = {'n_estimators': hp.quniform('n_estimators', 100, 4000, 200),
                                   'max_depth': hp.quniform('max_depth', 1, 10, 3),
                                   'learning_rate': hp.uniform('learning_rate', 1e-4, 1),
                                   'subsample': hp.uniform('subsample', 0.2, 1),
                                   'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
                                   'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1),
                                   'alpha': hp.choice('alpha',
                                                      [0, hp.uniform('alpha_', 1e-4, 1e2)]),
                                   'lambda': hp.choice('lambda',
                                                       [0, hp.uniform('lambda_', 1e-4, 1e2)]),
                                   'gamma': hp.choice('gamma',
                                                      [0, hp.uniform('gamma_', 1e-4, 1e2)])}
        best = fmin(fn=self.xgboost_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def xgboost_out_of_sample_test(self, best_hopt):
        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = XGBClassifier(n_estimators=int(best_hopt['n_estimators']),
                              max_depth=int(best_hopt['max_depth']),
                              learning_rate=best_hopt['learning_rate'],
                              subsample=best_hopt['subsample'],
                              colsample_bytree=best_hopt['colsample_bytree'],
                              colsample_bylevel=best_hopt['colsample_bylevel'],
                              gamma=best_hopt['gamma'],
                              alpha=best_hopt['alpha'],
                              lambda_=best_hopt['lambda'],
                              n_jobs=8)

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs

    def xgboost_manager(self):
        root_folder = f'./Homological_FS/XGBoostClassifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.xgboost_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')

        preds, probs = self.xgboost_out_of_sample_test(best_hopt=best_hopt)
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === CatBoost Optimization === #

    def cat_boost_objective(self, optimization_parameters):
        number_estimators = int(optimization_parameters['n_estimators'])
        learning_rate = optimization_parameters['learning_rate']
        random_strength = int(optimization_parameters['random_strength'])
        max_depth = int(optimization_parameters['max_depth'])
        l2_leaf_reg = optimization_parameters['l2_leaf_reg']
        bagging_temperature = optimization_parameters['bagging_temperature']
        leaf_estimation_iterations = int(optimization_parameters['leaf_estimation_iterations'])

        model = CatBoostClassifier(n_estimators=number_estimators,
                                   learning_rate=learning_rate,
                                   random_strength=random_strength,
                                   max_depth=max_depth,
                                   l2_leaf_reg=l2_leaf_reg,
                                   bagging_temperature=bagging_temperature,
                                   leaf_estimation_iterations=leaf_estimation_iterations,
                                   allow_writing_files=False,
                                   verbose=False)

        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        try:
            model.fit(scaled_X_train, self.y_train)
            preds = model.predict(scaled_X_val)
            score = f1_score(self.y_val, preds, average='macro')
            return -score
        except:
            return 0

    @measure_execution_time
    def cat_boost_optimize(self, trial):
        optimization_parameters = {'n_estimators': hp.quniform('n_estimators', 100, 4000, 300),
                                   'max_depth': hp.quniform('max_depth', 1, 10, 3),
                                   'learning_rate': hp.uniform('learning_rate', 1e-4, 1),
                                   'random_strength': hp.quniform('random_strength', 1, 10, 3),
                                   'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 10, 3),
                                   'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                                   'leaf_estimation_iterations': hp.quniform('leaf_estimation_iterations', 1, 10, 3)
                                   }
        best = fmin(fn=self.cat_boost_objective,
                    space=optimization_parameters,
                    algo=tpe.suggest,
                    trials=trial,
                    max_evals=params.HOPT_MAX_ITERATIONS,
                    rstate=np.random.default_rng(self.seed))
        return best

    def cat_boost_out_of_sample_test(self, best_hopt):
        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = CatBoostClassifier(n_estimators=int(best_hopt['n_estimators']),
                                   max_depth=int(best_hopt['max_depth']),
                                   learning_rate=best_hopt['learning_rate'],
                                   random_strength=int(best_hopt['random_strength']),
                                   l2_leaf_reg=int(best_hopt['l2_leaf_reg']),
                                   bagging_temperature=best_hopt['bagging_temperature'],
                                   leaf_estimation_iterations=int(best_hopt['leaf_estimation_iterations']),
                                   allow_writing_files=False,
                                   verbose=False)

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs

    def cat_boost_manager(self):
        root_folder = f'./Homological_FS/CatBoostClassifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.cat_boost_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')

        preds, probs = self.cat_boost_out_of_sample_test(best_hopt=best_hopt)
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === LightGBM Optimization === #

    def lightgbm_objective(self, optimization_parameters):
        number_estimators = int(optimization_parameters['n_estimators'])
        learning_rate = optimization_parameters['learning_rate']
        num_leaves = int(optimization_parameters['num_leaves'])
        max_depth = int(optimization_parameters['max_depth'])
        reg_alpha = float(optimization_parameters['reg_alpha'])
        reg_lambda = float(optimization_parameters['reg_lambda'])
        subsample = optimization_parameters['subsample']

        model = LGBMClassifier(n_estimators=number_estimators,
                               learning_rate=learning_rate,
                               num_leaves=num_leaves,
                               max_depth=max_depth,
                               reg_alpha=reg_alpha,
                               reg_lambda=reg_lambda,
                               subsample=subsample)

        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        try:
            model.fit(scaled_X_train, self.y_train)
            preds = model.predict(scaled_X_val)
            score = f1_score(self.y_val, preds, average='macro')
            return -score
        except:
            return 0

    @measure_execution_time
    def lightgbm_optimize(self, trial, choices_dict):
        optimization_parameters = {'n_estimators': hp.quniform('n_estimators', 100, 4000, 300),
                                   'max_depth': hp.quniform('max_depth', 1, 10, 3),
                                   'learning_rate': hp.uniform('learning_rate', 1e-4, 1),
                                   'num_leaves': hp.quniform('num_leaves', 5, 50, 5),
                                   'reg_alpha': hp.choice('reg_alpha', choices_dict['reg_alpha']),
                                   'reg_lambda': hp.choice('reg_lambda', choices_dict['reg_lambda']),
                                   'subsample': hp.uniform('subsample', 0.2, 0.8)
                                   }

        best = fmin(fn=self.lightgbm_objective,
                    space=optimization_parameters,
                    algo=tpe.suggest,
                    trials=trial,
                    max_evals=params.HOPT_MAX_ITERATIONS,
                    rstate=np.random.default_rng(self.seed))
        return best

    def lightgbm_out_of_sample_test(self, best_hopt, choices_dict):
        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = LGBMClassifier(n_estimators=int(best_hopt['n_estimators']),
                               max_depth=int(best_hopt['max_depth']),
                               learning_rate=best_hopt['learning_rate'],
                               num_leaves=int(best_hopt['num_leaves']),
                               reg_alpha=best_hopt['reg_alpha'],
                               reg_lambda=best_hopt['reg_lambda'],
                               subsample=best_hopt['subsample'])

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs

    def lightgbm_manager(self):
        reg_alpha = [0, 0.01, 1, 2, 5, 7, 10, 50, 100]
        reg_lambda = [0, 0.01, 1, 5, 10, 20, 50, 100]
        choices_dict = {'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda}

        root_folder = f'./Homological_FS/LightGBM_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.lightgbm_optimize(trial_hopt, choices_dict=choices_dict)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')

        preds, probs = self.lightgbm_out_of_sample_test(best_hopt=best_hopt, choices_dict=choices_dict)
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === TabPFN Optimization === #

    def tab_pfn_objective(self, optimization_parameters):
        nec = int(optimization_parameters['N_ensemble_configurations'])

        device = torch.device(params.DEVICE if torch.cuda.is_available() else "cpu")
        print("Using device: ", device)

        model = TabPFNClassifier(device=device, N_ensemble_configurations=nec)

        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.y_train, overwrite_warning=True)
        preds = model.predict(scaled_X_val)
        score = f1_score(self.y_val, preds, average='macro')
        return -score

    @measure_execution_time
    def tab_pfn_optimize(self, trial):
        optimization_parameters = {'N_ensemble_configurations': hp.quniform('N_ensemble_configurations', 8, 128, 8)}
        best = fmin(fn=self.tab_pfn_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def tab_pfn_out_of_sample_test(self, best_hopt):
        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        device = torch.device(params.DEVICE if torch.cuda.is_available() else "cpu")
        print("Using device: ", device)

        model = TabPFNClassifier(device=device, N_ensemble_configurations=int(best_hopt['N_ensemble_configurations']))

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train, overwrite_warning=True)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs

    def tab_pfn_manager(self):
        root_folder = f'./Homological_FS/TabPFN_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.tab_pfn_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')

        preds, probs = self.tab_pfn_out_of_sample_test(best_hopt=best_hopt)
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === TabNet Optimization === #

    def tab_net_objective(self, optimization_parameters):
        learning_rate = optimization_parameters['learning_rate']
        n_steps = int(optimization_parameters['n_steps'])
        relaxation_factor = optimization_parameters['relaxation_factor']

        model = TabNetClassifier(optimizer_params=dict(lr=learning_rate),
                                 n_steps=n_steps,
                                 gamma=relaxation_factor,
                                 verbose=0,
                                 device_name='auto')

        np_y_train = np.array(self.y_train)
        np_y_val = np.array(self.y_val)

        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, np_y_train, eval_set=[(scaled_X_val, np_y_val)],
                  batch_size=params.BATCH_SIZE,
                  max_epochs=params.MAX_EPOCHS,
                  patience=params.EARLY_STOPPING_PATIENCE)
        preds = model.predict(scaled_X_val)
        score = f1_score(self.y_val, preds, average='macro')
        return -score

    @measure_execution_time
    def tab_net_optimize(self, trial):
        optimization_parameters = {'learning_rate': hp.uniform('learning_rate', 1e-4, 1),
                                   'n_steps': hp.quniform('n_steps', 1, 8, 1),
                                   'relaxation_factor': hp.uniform('relaxation_factor', 0.3, 2)}

        best = fmin(fn=self.tab_net_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def tab_net_out_of_sample_test(self, best_hopt):
        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = TabNetClassifier(optimizer_params=dict(lr=best_hopt['learning_rate']),
                                 n_steps=int(best_hopt['n_steps']),
                                 gamma=best_hopt['relaxation_factor'],
                                 verbose=0,
                                 device_name='auto')

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)
        _, scaled_X_val = scaling(X_train=self.X_train,
                                  X_additional=self.X_val,
                                  choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train,
                  eval_set=[(scaled_X_val, np.array(self.y_val))],
                  batch_size=params.BATCH_SIZE,
                  max_epochs=params.MAX_EPOCHS,
                  patience=params.EARLY_STOPPING_PATIENCE)

        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs

    def tab_net_manager(self):
        root_folder = f'./Homological_FS/TabNet_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.tab_net_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')

        preds, probs = self.tab_net_out_of_sample_test(best_hopt=best_hopt)
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === Homological Convolutional Neural Network Optimization === #

    def hcnn_net_objective(self, optimization_parameters):
        n_filters_l1 = int(optimization_parameters['n_filters_l1'])
        n_filters_l2 = int(optimization_parameters['n_filters_l2'])
        tmfg_iterations = int(optimization_parameters['tmfg_iterations'])
        tmfg_confidence = optimization_parameters['tmfg_confidence']
        tmfg_similarity = optimization_parameters['tmfg_similarity']

        local_X_train = self.X_train.copy()
        local_X_val = self.X_val.copy()
        local_X_test = self.X_test.copy()
        local_y_train = self.y_train.copy()
        local_y_val = self.y_val.copy()
        local_y_test = self.y_test.copy()

        model = HCNN(X_train=local_X_train,
                     X_val=local_X_val,
                     X_test=local_X_test,
                     y_train=local_y_train,
                     y_val=local_y_val,
                     y_test=local_y_test,
                     n_filters_l1=n_filters_l1,
                     n_filters_l2=n_filters_l2,
                     tmfg_repetitions=tmfg_iterations,
                     tmfg_confidence=tmfg_confidence,
                     tmfg_similarity=tmfg_similarity)

        model.data_preparation_pipeline()

        model.fit()
        targets, preds, probs = model.evaluate()
        score = f1_score(targets, preds, average='macro')
        return -score

    @measure_execution_time
    def hcnn_net_optimize(self, trial):
        optimization_parameters = {'n_filters_l1': hp.quniform('n_filters_l1', 4, 16, 4),
                                   'n_filters_l2': hp.quniform('n_filters_l2', 32, 64, 8),
                                   'tmfg_iterations': hp.quniform('tmfg_iterations', 100, 1000, 300),
                                   'tmfg_confidence': hp.choice('tmfg_confidence', self.tmfg_pvalues_options),
                                   'tmfg_similarity': hp.choice('tmfg_similarity', self.tmfg_similarities_options),
                                   }
        best = fmin(fn=self.hcnn_net_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def hcnn_out_of_sample_test(self, best_hopt):
        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        self.post_opt_X_train = np.array(self.post_opt_X_train)
        self.post_opt_y_train = np.array(self.post_opt_y_train)

        local_X_train = self.post_opt_X_train.copy()
        local_X_val = self.X_val.copy()
        local_X_test = self.X_test.copy()
        local_y_train = self.post_opt_y_train.copy()
        local_y_val = self.y_val.copy()
        local_y_test = self.y_test.copy()

        model = HCNN(X_train=local_X_train,
                     X_val=local_X_val,
                     X_test=local_X_test,
                     y_train=local_y_train,
                     y_val=local_y_val,
                     y_test=local_y_test,
                     n_filters_l1=int(best_hopt['n_filters_l1']),
                     n_filters_l2=int(best_hopt['n_filters_l2']),
                     tmfg_repetitions=int(best_hopt['tmfg_iterations']),
                     tmfg_confidence=best_hopt['tmfg_confidence'],
                     tmfg_similarity=self.tmfg_similarities_options[best_hopt['tmfg_similarity']]
                     )

        model.data_preparation_pipeline()
        model.fit()

        targets, preds, probs = model.predict()
        score = classification_report(self.y_test, preds)
        print(score)
        return None, preds, probs

    def hcnn_manager(self):
        root_folder = f'./Homological_FS/HCNN_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.hcnn_net_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)

        number_of_selected_features, preds, probs = self.hcnn_out_of_sample_test(best_hopt=best_hopt)
        best_hopt['number_of_selected_features'] = [number_of_selected_features]
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === Homological Random Forest Optimization === #

    def h_random_forest_objective(self, optimization_parameters):
        est = int(optimization_parameters['n_estimators'])
        md = int(optimization_parameters['max_depth'])
        msl = int(optimization_parameters['min_samples_leaf'])
        mss = int(optimization_parameters['min_samples_split'])
        tmfg_iterations = int(optimization_parameters['tmfg_iterations'])
        tmfg_confidence = int(optimization_parameters['tmfg_confidence'])
        tmfg_similarity = optimization_parameters['tmfg_similarity']

        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=tmfg_iterations,
                                     tmfg_confidence=tmfg_confidence,
                                     tmfg_similarity=tmfg_similarity,
                                     seed=self.seed)
        local_X_train, local_X_val, local_X_test, number_selected_features = hdm.get_homological_data()

        if local_X_train.shape[1] == 0:
            return 0

        model = RandomForestClassifier(n_estimators=est, max_depth=md, min_samples_leaf=msl, min_samples_split=mss)

        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        try:
            model.fit(scaled_X_train, self.y_train)
            preds = model.predict(scaled_X_val)
            score = f1_score(self.y_val, preds, average='macro')
            return -score
        except:
            return 0

    @measure_execution_time
    def h_random_forest_optimize(self, trial):
        optimization_parameters = {'n_estimators': hp.quniform('n_estimators', 100, 4000, 200),
                                   'max_depth': hp.quniform('max_depth', 10, 50, 10),
                                   'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 10, 2),
                                   'min_samples_split': hp.quniform('min_samples_split', 2, 10, 2),
                                   'tmfg_iterations': hp.quniform('tmfg_iterations', 100, 1000, 300),
                                   'tmfg_confidence': hp.quniform('tmfg_confidence', 85, 99, 3),
                                   'tmfg_similarity': hp.choice('tmfg_similarity', self.tmfg_similarities_options)}
        best = fmin(fn=self.h_random_forest_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def h_random_forest_out_of_sample_test(self, best_hopt):
        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=int(best_hopt['tmfg_iterations']),
                                     tmfg_confidence=int(best_hopt['tmfg_confidence']),
                                     tmfg_similarity=self.tmfg_similarities_options[int(best_hopt['tmfg_similarity'])],
                                     seed=self.seed)
        self.X_train, self.X_val, self.X_test, number_selected_features = hdm.get_homological_data()

        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = RandomForestClassifier(n_estimators=int(best_hopt['n_estimators']),
                                       max_depth=int(best_hopt['max_depth']),
                                       min_samples_leaf=int(best_hopt['min_samples_leaf']),
                                       min_samples_split=int(best_hopt['min_samples_split']))

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs, number_selected_features

    def h_random_forest_manager(self):
        root_folder = f'./Homological_FS/HRandomForest_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.h_random_forest_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)

        preds, probs, number_selected_features = self.h_random_forest_out_of_sample_test(best_hopt=best_hopt)
        best_hopt['number_selected_features'] = [number_selected_features]
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === Homological XGBoost Optimization=== #

    def h_xgboost_objective(self, optimization_parameters):
        learning_rate = optimization_parameters['learning_rate']
        max_dept = int(optimization_parameters['max_depth'])
        n_estimators = int(optimization_parameters['n_estimators'])
        subsample = optimization_parameters['subsample']
        colsample_bytree = optimization_parameters['colsample_bytree']
        colsample_bylevel = optimization_parameters['colsample_bylevel']
        gamma = optimization_parameters['gamma']
        alpha = optimization_parameters['alpha']
        lambda_ = optimization_parameters['lambda']
        tmfg_iterations = int(optimization_parameters['tmfg_iterations'])
        tmfg_confidence = int(optimization_parameters['tmfg_confidence'])
        tmfg_similarity = optimization_parameters['tmfg_similarity']

        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=tmfg_iterations,
                                     tmfg_confidence=tmfg_confidence,
                                     tmfg_similarity=tmfg_similarity,
                                     seed=self.seed)
        local_X_train, local_X_val, local_X_test, number_selected_features = hdm.get_homological_data()

        if local_X_train.shape[1] == 0:
            return 0

        model = XGBClassifier(n_estimators=n_estimators,
                              max_depth=max_dept,
                              learning_rate=learning_rate,
                              subsample=subsample, colsample_bytree=colsample_bytree,
                              colsample_bylevel=colsample_bylevel,
                              gamma=gamma, alpha=alpha, lambda_=lambda_,
                              n_jobs=8)
        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        try:
            model.fit(scaled_X_train, self.y_train)
            preds = model.predict(scaled_X_val)
            score = f1_score(self.y_val, preds, average='macro')
            return -score
        except:
            return 0

    @measure_execution_time
    def h_xgboost_optimize(self, trial):
        optimization_parameters = {'n_estimators': hp.quniform('n_estimators', 100, 4000, 300),
                                   'max_depth': hp.quniform('max_depth', 1, 10, 3),
                                   'learning_rate': hp.uniform('learning_rate', 1e-4, 1),
                                   'subsample': hp.uniform('subsample', 0.2, 1),
                                   'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
                                   'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1),
                                   'alpha': hp.choice('alpha',
                                                      [0, hp.uniform('alpha_', 1e-4, 1e2)]),
                                   'lambda': hp.choice('lambda',
                                                       [0, hp.uniform('lambda_', 1-4, 1e2)]),
                                   'gamma': hp.choice('gamma',
                                                      [0, hp.uniform('gamma_', 1e-4, 1e2)]),
                                   'tmfg_iterations': hp.quniform('tmfg_iterations', 100, 1000, 300),
                                   'tmfg_confidence': hp.quniform('tmfg_confidence', 85, 99, 3),
                                   'tmfg_similarity': hp.choice('tmfg_similarity', self.tmfg_similarities_options)}
        best = fmin(fn=self.h_xgboost_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def h_xgboost_out_of_sample_test(self, best_hopt):
        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=int(best_hopt['tmfg_iterations']),
                                     tmfg_confidence=int(best_hopt['tmfg_confidence']),
                                     tmfg_similarity=self.tmfg_similarities_options[int(best_hopt['tmfg_similarity'])],
                                     seed=self.seed)
        self.X_train, self.X_val, self.X_test, number_selected_features = hdm.get_homological_data()

        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = XGBClassifier(n_estimators=int(best_hopt['n_estimators']),
                              max_depth=int(best_hopt['max_depth']),
                              learning_rate=best_hopt['learning_rate'],
                              subsample=best_hopt['subsample'],
                              colsample_bytree=best_hopt['colsample_bytree'],
                              colsample_bylevel=best_hopt['colsample_bylevel'],
                              gamma=best_hopt['gamma'],
                              alpha=best_hopt['alpha'],
                              lambda_=best_hopt['lambda'],
                              n_jobs=8)

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs, number_selected_features

    def h_xgboost_manager(self):
        root_folder = f'./Homological_FS/HXGBoost_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.h_xgboost_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)

        preds, probs, number_selected_features = self.h_xgboost_out_of_sample_test(best_hopt=best_hopt)
        best_hopt['number_selected_features'] = [number_selected_features]
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === Homological TabPFN Optimization === #

    def h_tab_pfn_objective(self, optimization_parameters):
        nec = int(optimization_parameters['N_ensemble_configurations'])
        tmfg_iterations = int(optimization_parameters['tmfg_iterations'])
        tmfg_confidence = int(optimization_parameters['tmfg_confidence'])
        tmfg_similarity = optimization_parameters['tmfg_similarity']

        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=tmfg_iterations,
                                     tmfg_confidence=tmfg_confidence,
                                     tmfg_similarity=tmfg_similarity,
                                     seed=self.seed)
        local_X_train, local_X_val, local_X_test, number_selected_features = hdm.get_homological_data()

        if local_X_train.shape[1] == 0:
            return 0

        model = TabPFNClassifier(device='cpu', N_ensemble_configurations=nec)

        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        try:
            model.fit(scaled_X_train, self.y_train)
            preds = model.predict(scaled_X_val)
            score = f1_score(self.y_val, preds, average='macro')
            return -score
        except:
            return 0

    @measure_execution_time
    def h_tab_pfn_optimize(self, trial):
        optimization_parameters = {'N_ensemble_configurations': hp.quniform('N_ensemble_configurations', 8, 128, 8),
                                   'tmfg_iterations': hp.quniform('tmfg_iterations', 100, 1000, 300),
                                   'tmfg_confidence': hp.quniform('tmfg_confidence', 85, 99, 3),
                                   'tmfg_similarity': hp.choice('tmfg_similarity', self.tmfg_similarities_options)}
        best = fmin(fn=self.h_tab_pfn_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def h_tab_pfn_out_of_sample_test(self, best_hopt):
        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=int(best_hopt['tmfg_iterations']),
                                     tmfg_confidence=int(best_hopt['tmfg_confidence']),
                                     tmfg_similarity=self.tmfg_similarities_options[int(best_hopt['tmfg_similarity'])],
                                     seed=self.seed)
        self.X_train, self.X_val, self.X_test, number_selected_features = hdm.get_homological_data()

        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = TabPFNClassifier(device='cpu',
                                 N_ensemble_configurations=int(best_hopt['N_ensemble_configurations']))

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs, number_selected_features

    def h_tab_pfn_manager(self):
        root_folder = f'./Homological_FS/HTabPFN_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.h_tab_pfn_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)

        preds, probs, number_selected_features = self.h_tab_pfn_out_of_sample_test(best_hopt=best_hopt)
        best_hopt['number_selected_features'] = [number_selected_features]
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === Homological CatBoost Optimization === #

    def h_cat_boost_objective(self, optimization_parameters):
        number_estimators = int(optimization_parameters['n_estimators'])
        learning_rate = optimization_parameters['learning_rate']
        random_strength = optimization_parameters['random_strength']
        max_depth = optimization_parameters['max_depth']
        l2_leaf_reg = optimization_parameters['l2_leaf_reg']
        bagging_temperature = optimization_parameters['bagging_temperature']
        leaf_estimation_iterations = optimization_parameters['leaf_estimation_iterations']
        tmfg_iterations = int(optimization_parameters['tmfg_iterations'])
        tmfg_confidence = int(optimization_parameters['tmfg_confidence'])
        tmfg_similarity = optimization_parameters['tmfg_similarity']

        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=tmfg_iterations,
                                     tmfg_confidence=tmfg_confidence,
                                     tmfg_similarity=tmfg_similarity,
                                     seed=self.seed)
        local_X_train, local_X_val, local_X_test, number_selected_features = hdm.get_homological_data()

        if local_X_train.shape[1] == 0:
            return 0

        model = CatBoostClassifier(n_estimators=number_estimators,
                                   learning_rate=learning_rate,
                                   random_strength=random_strength,
                                   max_depth=max_depth,
                                   l2_leaf_reg=l2_leaf_reg,
                                   bagging_temperature=bagging_temperature,
                                   leaf_estimation_iterations=leaf_estimation_iterations,
                                   allow_writing_files=False,
                                   verbose=False)

        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        try:
            model.fit(scaled_X_train, self.y_train)
            preds = model.predict(scaled_X_val)
            score = f1_score(self.y_val, preds, average='macro')
            return -score
        except:
            return 0

    @measure_execution_time
    def h_cat_boost_optimize(self, trial):
        optimization_parameters = {'n_estimators': hp.quniform('n_estimators', 100, 4000, 300),
                                   'max_depth': hp.quniform('max_depth', 1, 10, 3),
                                   'learning_rate': hp.uniform('learning_rate', 1e-4, 1),
                                   'random_strength': hp.quniform('random_strength', 1, 10, 3),
                                   'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 10, 3),
                                   'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                                   'leaf_estimation_iterations': hp.quniform('leaf_estimation_iterations', 1, 10, 3),
                                   'tmfg_iterations': hp.quniform('tmfg_iterations', 100, 1000, 300),
                                   'tmfg_confidence': hp.quniform('tmfg_confidence', 85, 99, 3),
                                   'tmfg_similarity': hp.choice('tmfg_similarity', self.tmfg_similarities_options)
                                   }
        best = fmin(fn=self.h_cat_boost_objective,
                    space=optimization_parameters,
                    algo=tpe.suggest,
                    trials=trial,
                    max_evals=params.HOPT_MAX_ITERATIONS,
                    rstate=np.random.default_rng(self.seed))
        return best

    def h_cat_boost_out_of_sample_test(self, best_hopt):
        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=int(best_hopt['tmfg_iterations']),
                                     tmfg_confidence=int(best_hopt['tmfg_confidence']),
                                     tmfg_similarity=self.tmfg_similarities_options[int(best_hopt['tmfg_similarity'])],
                                     seed=self.seed)
        self.X_train, self.X_val, self.X_test, number_selected_features = hdm.get_homological_data()

        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = CatBoostClassifier(n_estimators=int(best_hopt['n_estimators']),
                                   max_depth=int(best_hopt['max_depth']),
                                   learning_rate=best_hopt['learning_rate'],
                                   random_strength=int(best_hopt['random_strength']),
                                   l2_leaf_reg=int(best_hopt['l2_leaf_reg']),
                                   bagging_temperature=best_hopt['bagging_temperature'],
                                   leaf_estimation_iterations=int(best_hopt['leaf_estimation_iterations']),
                                   allow_writing_files=False,
                                   verbose=False)

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs, number_selected_features

    def h_cat_boost_manager(self):
        root_folder = f'./Homological_FS/HCatBoost_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.h_cat_boost_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)

        preds, probs, number_selected_features = self.h_cat_boost_out_of_sample_test(best_hopt=best_hopt)
        best_hopt['number_selected_features'] = [number_selected_features]
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === Homological LightGBM Optimization === #

    def h_lightgbm_objective(self, optimization_parameters):
        number_estimators = int(optimization_parameters['n_estimators'])
        learning_rate = optimization_parameters['learning_rate']
        num_leaves = int(optimization_parameters['num_leaves'])
        max_depth = int(optimization_parameters['max_depth'])
        reg_alpha = float(optimization_parameters['reg_alpha'])
        reg_lambda = float(optimization_parameters['reg_lambda'])
        subsample = optimization_parameters['subsample']
        tmfg_iterations = int(optimization_parameters['tmfg_iterations'])
        tmfg_confidence = int(optimization_parameters['tmfg_confidence'])
        tmfg_similarity = optimization_parameters['tmfg_similarity']

        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=tmfg_iterations,
                                     tmfg_confidence=tmfg_confidence,
                                     tmfg_similarity=tmfg_similarity,
                                     seed=self.seed)
        local_X_train, local_X_val, local_X_test, number_selected_features  = hdm.get_homological_data()

        if local_X_train.shape[1] == 0:
            return 0

        model = LGBMClassifier(n_estimators=number_estimators,
                               learning_rate=learning_rate,
                               num_leaves=num_leaves,
                               max_depth=max_depth,
                               reg_alpha=reg_alpha,
                               reg_lambda=reg_lambda,
                               subsample=subsample)

        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        try:
            model.fit(scaled_X_train, self.y_train)
            preds = model.predict(scaled_X_val)
            score = f1_score(self.y_val, preds, average='macro')
            return -score
        except:
            return 0

    @measure_execution_time
    def h_lightgbm_optimize(self, trial, choices_dict):
        optimization_parameters = {'n_estimators': hp.quniform('n_estimators', 100, 4000, 300),
                                   'max_depth': hp.quniform('max_depth', 1, 10, 3),
                                   'learning_rate': hp.uniform('learning_rate', 1e-4, 1),
                                   'num_leaves': hp.quniform('num_leaves', 5, 50, 5),
                                   'reg_alpha': hp.choice('reg_alpha', choices_dict['reg_alpha']),
                                   'reg_lambda': hp.choice('reg_lambda', choices_dict['reg_lambda']),
                                   'subsample': hp.uniform('subsample', 0.2, 0.8),
                                   'tmfg_iterations': hp.quniform('tmfg_iterations', 100, 1000, 300),
                                   'tmfg_confidence': hp.quniform('tmfg_confidence', 85, 99, 3),
                                   'tmfg_similarity': hp.choice('tmfg_similarity', self.tmfg_similarities_options)
                                   }

        best = fmin(fn=self.h_lightgbm_objective,
                    space=optimization_parameters,
                    algo=tpe.suggest,
                    trials=trial,
                    max_evals=params.HOPT_MAX_ITERATIONS,
                    rstate=np.random.default_rng(self.seed))
        return best

    def h_lightgbm_out_of_sample_test(self, best_hopt, choices_dict):
        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=int(best_hopt['tmfg_iterations']),
                                     tmfg_confidence=int(best_hopt['tmfg_confidence']),
                                     tmfg_similarity=self.tmfg_similarities_options[int(best_hopt['tmfg_similarity'])],
                                     seed=self.seed)
        self.X_train, self.X_val, self.X_test, number_selected_features = hdm.get_homological_data()

        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = LGBMClassifier(n_estimators=int(best_hopt['n_estimators']),
                               max_depth=int(best_hopt['max_depth']),
                               learning_rate=best_hopt['learning_rate'],
                               num_leaves=int(best_hopt['num_leaves']),
                               reg_alpha=best_hopt['reg_alpha'],
                               reg_lambda=best_hopt['reg_lambda'],
                               subsample=best_hopt['subsample'])

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs, number_selected_features

    def h_lightgbm_manager(self):
        reg_alpha = [0, 0.01, 1, 2, 5, 7, 10, 50, 100]
        reg_lambda = [0, 0.01, 1, 5, 10, 20, 50, 100]

        choices_dict = {'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda}

        root_folder = f'./Homological_FS/HLightGBM_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.h_lightgbm_optimize(trial_hopt, choices_dict=choices_dict)
        best_hopt = update_keys(d=best_hopt)

        preds, probs, number_selected_features = self.h_lightgbm_out_of_sample_test(best_hopt=best_hopt, choices_dict=choices_dict)
        best_hopt['number_selected_features'] = [number_selected_features]
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')

    # === Homological TabNet Optimization ===

    def h_tab_net_objective(self, optimization_parameters):
        learning_rate = optimization_parameters['learning_rate']
        n_steps = int(optimization_parameters['n_steps'])
        relaxation_factor = optimization_parameters['relaxation_factor']
        tmfg_iterations = int(optimization_parameters['tmfg_iterations'])
        tmfg_confidence = int(optimization_parameters['tmfg_confidence'])
        tmfg_similarity = optimization_parameters['tmfg_similarity']

        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=tmfg_iterations,
                                     tmfg_confidence=tmfg_confidence,
                                     tmfg_similarity=tmfg_similarity,
                                     seed=self.seed)
        local_X_train, local_X_val, local_X_test, number_selected_features = hdm.get_homological_data()

        if local_X_train.shape[1] == 0:
            return 0

        model = TabNetClassifier(optimizer_params=dict(lr=learning_rate),
                                 n_steps=n_steps,
                                 gamma=relaxation_factor,
                                 verbose=0)

        np_y_train = np.array(self.y_train)
        np_y_val = np.array(self.y_val)

        scaled_X_train, scaled_X_val = scaling(X_train=self.X_train,
                                               X_additional=self.X_val,
                                               choice=params.SCALING_SCHEME)

        try:
            model.fit(scaled_X_train, np_y_train, eval_set=[(scaled_X_val, np_y_val)], batch_size=16)
            preds = model.predict(scaled_X_val)
            score = f1_score(self.y_val, preds, average='macro')
            return -score
        except:
            return 0

    @measure_execution_time
    def h_tab_net_optimize(self, trial):
        optimization_parameters = {'learning_rate': hp.uniform('learning_rate', 1e-4, 1),
                                   'n_steps': hp.quniform('n_steps', 1, 8, 1),
                                   'relaxation_factor': hp.uniform('relaxation_factor', 0.3, 2),
                                   'tmfg_iterations': hp.quniform('tmfg_iterations', 100, 1000, 300),
                                   'tmfg_confidence': hp.quniform('tmfg_confidence', 85, 99, 3),
                                   'tmfg_similarity': hp.choice('tmfg_similarity', self.tmfg_similarities_options)
                                   }

        best = fmin(fn=self.h_tab_net_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def h_tab_net_out_of_sample_test(self, best_hopt):
        hdm = HomologicalDataManager(X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
                                     tmfg_iterations=int(best_hopt['tmfg_iterations']),
                                     tmfg_confidence=int(best_hopt['tmfg_confidence']),
                                     tmfg_similarity=self.tmfg_similarities_options[int(best_hopt['tmfg_similarity'])],
                                     seed=self.seed)
        self.X_train, self.X_val, self.X_test, number_selected_features = hdm.get_homological_data()

        self.post_opt_X_train = pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = TabNetClassifier(optimizer_params=dict(lr=best_hopt['learning_rate']),
                                 n_steps=int(best_hopt['n_steps']),
                                 gamma=best_hopt['relaxation_factor'],
                                 verbose=0)

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)
        _, scaled_X_val = scaling(X_train=self.X_train,
                                  X_additional=self.X_val,
                                  choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train,
                  eval_set=[(scaled_X_val, np.array(self.y_val))],
                  batch_size=params.BATCH_SIZE)

        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)
        score = classification_report(self.y_test, preds)
        print(score)
        return preds, probs, number_selected_features

    def h_tab_net_manager(self):
        root_folder = f'./Homological_FS/HTabNet_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(root_folder)

        trial_hopt = Trials()
        best_hopt = self.h_tab_net_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)

        preds, probs, number_selected_features = self.h_tab_net_out_of_sample_test(best_hopt=best_hopt)
        best_hopt['number_selected_features'] = [number_selected_features]
        write_json_file(best_hopt, root_folder + 'best_hopt.csv')
        merge_probs_preds_classification(probs, preds, self.y_test, root_folder + 'pobs_preds.csv')
