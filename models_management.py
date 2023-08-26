from catboost import CatBoostClassifier
from hyperopt import tpe, hp, Trials
from hyperopt.fmin import fmin
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from HCNN import *
from data_management import *
from models_utils import *

import shutil


class ModelsManager:
    def __init__(self, model, seed, dataset_id, X_train, X_val, X_test, y_train, y_val, y_test):
        self.model = model
        self.seed = seed
        self.dataset_id = dataset_id

        self.dataset_time_mapping = {'11': 3206, '1462': 6995, '1464': 1843, '18': 7628, '37': 1631, '15': 1686,
                                     '54': 4031, '40994': 1875, '1063': 2302, '1068': 3691, '40982': 6231, '1510': 2761,
                                     '1049': 6442, '1050': 4832, '1494': 3717, '22': 5883, '16': 7347, '458': 6281, '14': 6401}

        ### === Fix seed === ###
        np.random.seed(self.seed)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train.ravel()
        self.y_val = y_val.ravel()
        self.y_test = y_test.ravel()

        self.root_folder = None

        self.available_models = ['RandomForest',
                                 'LogisticRegression',
                                 'XGBoost',
                                 'CatBoost',
                                 'LightGBM',
                                 'MLP',
                                 'TabularTransformer',
                                 'TabNet',
                                 'HCNN']
        assert self.model in self.available_models, 'Model not recognised'

        self.post_opt_X_train = None
        self.post_opt_y_train = None

        self.tmfg_similarities_options = ['pearson', 'spearman']
        #self.tmfg_pvalues_options = [90, 95, 99]
        #self.filtering_options = ['TMFG_Bootstrapping']
        self.tmfg_pvalues_options = [0] #TODO: uncomment
        self.filtering_options = ['TMFG_StatMatrix'] #TODO: uncomment
        self.dropout_options = [0.25]

    # === Logistic Regression Optimization === #

    def logistic_regression_objective(self, optimization_parameters):
        penalty = str(optimization_parameters['penalty'])
        max_iter = int(optimization_parameters['max_iter'])

        model = LogisticRegression(penalty=penalty,
                                   max_iter=max_iter)

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
    def logistic_regression_optimize(self, trial, choices_dict):
        optimization_parameters = {'penalty': hp.choice('penalty', choices_dict['penalty']),
                                   'max_iter': hp.choice('max_iter', choices_dict['max_iter'])}
        best = fmin(fn=self.logistic_regression_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def logistic_regression_out_of_sample_test(self, best_hopt, choices_dict):
        oos_time_start = time.time()

        self.post_opt_X_train = self.X_train  # pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = self.y_train  # pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = LogisticRegression(penalty=str(choices_dict['penalty'][best_hopt['penalty']]),
                                   max_iter=int(choices_dict['max_iter'][best_hopt['max_iter']]))

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)

        oos_time_end = time.time()

        score = classification_report(self.y_test, preds)
        print(score)

        oos_time = oos_time_end - oos_time_start
        return preds, probs, oos_time

    def logistic_regression_manager(self):
        penalty = ['l1', 'l2', 'elasticnet']
        max_iter = ['100', '500', '1000']
        choices_dict = {'penalty': penalty, 'max_iter': max_iter}

        self.root_folder = f'./Homological_FS/LogisticRegressionClassifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(self.root_folder)

        trial_hopt = Trials()
        best_hopt = self.logistic_regression_optimize(trial_hopt, choices_dict)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, self.root_folder + 'best_hopt.csv')

        preds, probs, oos_time = self.logistic_regression_out_of_sample_test(best_hopt=best_hopt, choices_dict=choices_dict)
        merge_probs_preds_classification(probs, preds, self.y_test, self.root_folder + 'pobs_preds.csv')

        oos_time_execution = pd.DataFrame({'time': [oos_time]})
        oos_time_execution.to_csv(self.root_folder + f'ofs_execution_time.csv', index=False)

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
        oos_time_start = time.time()

        self.post_opt_X_train = self.X_train #pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = self.y_train #pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
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

        oos_time_end = time.time()

        score = classification_report(self.y_test, preds)
        print(score)
        oos_time = oos_time_end - oos_time_start
        return preds, probs, oos_time

    def random_forest_manager(self):
        self.root_folder = f'./Homological_FS/RandomForestClassifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(self.root_folder)

        trial_hopt = Trials()
        best_hopt = self.random_forest_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, self.root_folder + 'best_hopt.csv')

        preds, probs, oos_time = self.random_forest_out_of_sample_test(best_hopt=best_hopt)
        merge_probs_preds_classification(probs, preds, self.y_test, self.root_folder + 'pobs_preds.csv')

        oos_time_execution = pd.DataFrame({'time': [oos_time]})
        oos_time_execution.to_csv(self.root_folder + f'ofs_execution_time.csv', index=False)

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

        if params.DEVICE == 'cuda':
            model = XGBClassifier(n_estimators=n_estimators,
                                  max_depth=max_dept,
                                  learning_rate=learning_rate,
                                  subsample=subsample, colsample_bytree=colsample_bytree,
                                  colsample_bylevel=colsample_bylevel,
                                  gamma=gamma, alpha=alpha, lambda_=lambda_,
                                  n_jobs=8, tree_method='gpu_hist')
        else:
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
        oos_time_start = time.time()

        self.post_opt_X_train = self.X_train #pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = self.y_train #pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        if params.DEVICE == 'cuda':
            model = XGBClassifier(n_estimators=int(best_hopt['n_estimators']),
                                  max_depth=int(best_hopt['max_depth']),
                                  learning_rate=best_hopt['learning_rate'],
                                  subsample=best_hopt['subsample'],
                                  colsample_bytree=best_hopt['colsample_bytree'],
                                  colsample_bylevel=best_hopt['colsample_bylevel'],
                                  gamma=best_hopt['gamma'],
                                  alpha=best_hopt['alpha'],
                                  lambda_=best_hopt['lambda'],
                                  n_jobs=8, tree_method='gpu_hist')
        else:
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

        oos_time_end = time.time()

        score = classification_report(self.y_test, preds)
        print(score)

        oos_time = oos_time_end - oos_time_start
        return preds, probs, oos_time

    def xgboost_manager(self):
        self.root_folder = f'./Homological_FS/XGBoostClassifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(self.root_folder)

        trial_hopt = Trials()
        best_hopt = self.xgboost_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, self.root_folder + 'best_hopt.csv')

        preds, probs, oos_time = self.xgboost_out_of_sample_test(best_hopt=best_hopt)
        merge_probs_preds_classification(probs, preds, self.y_test, self.root_folder + 'pobs_preds.csv')

        oos_time_execution = pd.DataFrame({'time': [oos_time]})
        oos_time_execution.to_csv(self.root_folder + f'ofs_execution_time.csv', index=False)

    # === CatBoost Optimization === #

    def cat_boost_objective(self, optimization_parameters):
        number_estimators = int(optimization_parameters['n_estimators'])
        learning_rate = optimization_parameters['learning_rate']
        random_strength = int(optimization_parameters['random_strength'])
        max_depth = int(optimization_parameters['max_depth'])
        l2_leaf_reg = optimization_parameters['l2_leaf_reg']
        bagging_temperature = optimization_parameters['bagging_temperature']
        leaf_estimation_iterations = int(optimization_parameters['leaf_estimation_iterations'])

        if params.DEVICE == 'cuda':
            model = CatBoostClassifier(n_estimators=number_estimators,
                                       learning_rate=learning_rate,
                                       random_strength=random_strength,
                                       max_depth=max_depth,
                                       l2_leaf_reg=l2_leaf_reg,
                                       bagging_temperature=bagging_temperature,
                                       leaf_estimation_iterations=leaf_estimation_iterations,
                                       allow_writing_files=False,
                                       verbose=False,
                                       task_type='GPU')
        else:
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
        oos_time_start = time.time()

        self.post_opt_X_train = self.X_train #pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = self.y_train #pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        if params.DEVICE == 'cuda':
            model = CatBoostClassifier(n_estimators=int(best_hopt['n_estimators']),
                                       max_depth=int(best_hopt['max_depth']),
                                       learning_rate=best_hopt['learning_rate'],
                                       random_strength=int(best_hopt['random_strength']),
                                       l2_leaf_reg=int(best_hopt['l2_leaf_reg']),
                                       bagging_temperature=best_hopt['bagging_temperature'],
                                       leaf_estimation_iterations=int(best_hopt['leaf_estimation_iterations']),
                                       allow_writing_files=False,
                                       verbose=False,
                                       task_type='GPU')
        else:
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

        oos_time_end = time.time()

        score = classification_report(self.y_test, preds)
        print(score)

        oos_time = oos_time_end - oos_time_start
        return preds, probs, oos_time

    def cat_boost_manager(self):
        self.root_folder = f'./Homological_FS/CatBoostClassifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(self.root_folder)

        trial_hopt = Trials()
        best_hopt = self.cat_boost_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, self.root_folder + 'best_hopt.csv')

        preds, probs, oos_time = self.cat_boost_out_of_sample_test(best_hopt=best_hopt)
        merge_probs_preds_classification(probs, preds, self.y_test, self.root_folder + 'pobs_preds.csv')

        oos_time_execution = pd.DataFrame({'time': [oos_time]})
        oos_time_execution.to_csv(self.root_folder + f'ofs_execution_time.csv', index=False)

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
        oos_time_start = time.time()

        self.post_opt_X_train = self.X_train #pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = self.y_train #pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = LGBMClassifier(n_estimators=int(best_hopt['n_estimators']),
                               max_depth=int(best_hopt['max_depth']),
                               learning_rate=best_hopt['learning_rate'],
                               num_leaves=int(best_hopt['num_leaves']),
                               reg_alpha=choices_dict['reg_alpha'][best_hopt['reg_alpha']],
                               reg_lambda=choices_dict['reg_lambda'][best_hopt['reg_lambda']],
                               subsample=best_hopt['subsample'])

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)

        oos_time_end = time.time()

        score = classification_report(self.y_test, preds)
        print(score)

        oos_time = oos_time_end - oos_time_start
        return preds, probs, oos_time

    def lightgbm_manager(self):
        reg_alpha = [0, 0.01, 1, 2, 5, 7, 10, 50, 100]
        reg_lambda = [0, 0.01, 1, 5, 10, 20, 50, 100]
        choices_dict = {'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda}

        self.root_folder = f'./Homological_FS/LightGBM_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(self.root_folder)

        trial_hopt = Trials()
        best_hopt = self.lightgbm_optimize(trial_hopt, choices_dict=choices_dict)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, self.root_folder + 'best_hopt.csv')

        preds, probs, oos_time = self.lightgbm_out_of_sample_test(best_hopt=best_hopt, choices_dict=choices_dict)
        merge_probs_preds_classification(probs, preds, self.y_test, self.root_folder + 'pobs_preds.csv')

        oos_time_execution = pd.DataFrame({'time': [oos_time]})
        oos_time_execution.to_csv(self.root_folder + f'ofs_execution_time.csv', index=False)

    # === MLPClassifier Regression Optimization === #

    def mlp_classifier_objective(self, optimization_parameters):
        model = MLPClassifier(hidden_layer_sizes=int(optimization_parameters['hidden_layer_sizes']),
                              alpha=float(optimization_parameters['alpha']),
                              max_iter=int(optimization_parameters['max_iter']),
                              learning_rate=str(optimization_parameters['learning_rate']),
                              random_state=self.seed,
                              early_stopping=True,
                              validation_fraction=0.1)

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
    def mlp_classifier_optimize(self, trial, choices_dict):
        optimization_parameters = {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', choices_dict['hidden_layer_sizes']),
                                   'alpha': hp.choice('alpha', choices_dict['alpha']),
                                   'max_iter': hp.choice('max_iter', choices_dict['max_iter']),
                                   'learning_rate': hp.choice('learning_rate', choices_dict['learning_rate'])}

        best = fmin(fn=self.mlp_classifier_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def mlp_classifier_out_of_sample_test(self, best_hopt, choices_dict):
        oos_time_start = time.time()

        self.post_opt_X_train = self.X_train  # pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = self.y_train  # pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        model = MLPClassifier(hidden_layer_sizes=int(choices_dict['hidden_layer_sizes'][best_hopt['hidden_layer_sizes']]),
                              alpha=float(choices_dict['alpha'][best_hopt['alpha']]),
                              max_iter=int(choices_dict['max_iter'][best_hopt['max_iter']]),
                              learning_rate=str(choices_dict['learning_rate'][best_hopt['learning_rate']]),
                              random_state=self.seed,
                              early_stopping=True,
                              validation_fraction=0.1)

        scaled_X_train, scaled_X_test = scaling(X_train=self.post_opt_X_train,
                                                X_additional=self.X_test,
                                                choice=params.SCALING_SCHEME)

        model.fit(scaled_X_train, self.post_opt_y_train)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)

        oos_time_end = time.time()

        score = classification_report(self.y_test, preds)
        print(score)

        oos_time = oos_time_end - oos_time_start
        return preds, probs, oos_time

    def mlp_manager(self):
        hidden_layer_sizes = [10, 50, 100, 150, 200]
        alpha = [0.1, 0.01, 0.001, 0.0001]
        max_iter = [100, 500, 1000]
        learning_rate = ['constant', 'invscaling', 'adaptive']

        choices_dict = {'hidden_layer_sizes': hidden_layer_sizes, 'alpha': alpha, 'max_iter': max_iter, 'learning_rate': learning_rate}

        self.root_folder = f'./Homological_FS/MLPClassifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(self.root_folder)

        trial_hopt = Trials()
        best_hopt = self.mlp_classifier_optimize(trial_hopt, choices_dict)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, self.root_folder + 'best_hopt.csv')

        preds, probs, oos_time = self.mlp_classifier_out_of_sample_test(best_hopt=best_hopt, choices_dict=choices_dict)
        merge_probs_preds_classification(probs, preds, self.y_test, self.root_folder + 'pobs_preds.csv')

        oos_time_execution = pd.DataFrame({'time': [oos_time]})
        oos_time_execution.to_csv(self.root_folder + f'ofs_execution_time.csv', index=False)


    # === TabPFN Optimization === #

    def tab_pfn_objective(self, optimization_parameters):
        nec = int(optimization_parameters['N_ensemble_configurations'])

        device = torch.device(params.DEVICE if torch.cuda.is_available() else "cpu")

        model = TabPFNClassifier(device=device, N_ensemble_configurations=nec)

        scaled_X_train = self.X_train
        scaled_X_val = self.X_val

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
        oos_time_start = time.time()

        self.post_opt_X_train = self.X_train #pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = self.y_train #pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        device = torch.device(params.DEVICE if torch.cuda.is_available() else "cpu")

        model = TabPFNClassifier(device=device, N_ensemble_configurations=int(best_hopt['N_ensemble_configurations']))

        scaled_X_train = self.post_opt_X_train
        scaled_X_test = self.X_test

        model.fit(scaled_X_train, self.post_opt_y_train, overwrite_warning=True)
        preds = model.predict(scaled_X_test)
        probs = model.predict_proba(scaled_X_test)

        oos_time_end = time.time()

        score = classification_report(self.y_test, preds)
        print(score)

        oos_time = oos_time_end - oos_time_start
        return preds, probs, oos_time

    def tab_pfn_manager(self):
        self.root_folder = f'./Homological_FS/TabPFN_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(self.root_folder)

        trial_hopt = Trials()
        best_hopt = self.tab_pfn_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, self.root_folder + 'best_hopt.csv')

        preds, probs, oos_time = self.tab_pfn_out_of_sample_test(best_hopt=best_hopt)
        merge_probs_preds_classification(probs, preds, self.y_test, self.root_folder + 'pobs_preds.csv')

        oos_time_execution = pd.DataFrame({'time': [oos_time]})
        oos_time_execution.to_csv(self.root_folder + f'ofs_execution_time.csv', index=False)

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
        oos_time_start = time.time()

        self.post_opt_X_train = self.X_train #pd.concat([self.X_train, self.X_val])
        self.post_opt_y_train = self.y_train #pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
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

        oos_time_end = time.time()
        score = classification_report(self.y_test, preds)
        print(score)

        oos_time = oos_time_end - oos_time_start
        return preds, probs, oos_time

    def tab_net_manager(self):
        self.root_folder = f'./Homological_FS/TabNet_Classifier/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(self.root_folder)

        trial_hopt = Trials()
        best_hopt = self.tab_net_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)
        write_json_file(best_hopt, self.root_folder + 'best_hopt.csv')

        preds, probs, oos_time = self.tab_net_out_of_sample_test(best_hopt=best_hopt)
        merge_probs_preds_classification(probs, preds, self.y_test, self.root_folder + 'pobs_preds.csv')

        oos_time_execution = pd.DataFrame({'time': [oos_time]})
        oos_time_execution.to_csv(self.root_folder + f'ofs_execution_time.csv', index=False)

    # === Homological Convolutional Neural Network Optimization === #

    def hcnn_net_objective(self, optimization_parameters):

        n_filters_l1 = int(optimization_parameters['n_filters_l1'])
        n_filters_l2 = int(optimization_parameters['n_filters_l2'])
        tmfg_iterations = int(optimization_parameters['tmfg_iterations'])
        tmfg_confidence = optimization_parameters['tmfg_confidence']
        tmfg_similarity = optimization_parameters['tmfg_similarity']
        filtering_type = optimization_parameters['filtering_type']
        dropout_rate = optimization_parameters['dropout_rate']

        local_X_train = self.X_train.copy()
        local_X_val = self.X_val.copy()
        local_X_test = self.X_test.copy()
        local_y_train = self.y_train.copy()
        local_y_val = self.y_val.copy()
        local_y_test = self.y_test.copy()

        try:
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
                         tmfg_similarity=tmfg_similarity,
                         root_folder=self.root_folder+'hyperopt/',
                         filtering_type=filtering_type,
                         dropout_rate=dropout_rate,
                         seed=self.seed,
                         )

            model.data_preparation_pipeline()

            model.fit()
            targets, preds, probs = model.evaluate()
            score = f1_score(targets, preds, average='macro')
            return -score
        except:
            return 0

    @measure_execution_time
    def hcnn_net_optimize(self, trial):
        optimization_parameters = {'n_filters_l1': hp.quniform('n_filters_l1', 4, 16, 4),
                                   'n_filters_l2': hp.quniform('n_filters_l2', 32, 64, 4),
                                   'tmfg_iterations': hp.quniform('tmfg_iterations', 100, 1000, 300),
                                   'tmfg_confidence': hp.choice('tmfg_confidence', self.tmfg_pvalues_options),
                                   'tmfg_similarity': hp.choice('tmfg_similarity', self.tmfg_similarities_options),
                                   'filtering_type': hp.choice('filtering_type', self.filtering_options),
                                   'dropout_rate': hp.choice('dropout_rate', self.dropout_options),
                                   }
        best = fmin(fn=self.hcnn_net_objective, space=optimization_parameters, algo=tpe.suggest,
                    trials=trial, max_evals=params.HOPT_MAX_ITERATIONS, rstate=np.random.default_rng(self.seed))
        return best

    def hcnn_out_of_sample_test(self, best_hopt):
        oos_time_start = time.time()

        self.post_opt_X_train = self.X_train
        self.post_opt_y_train = self.y_train
        self.post_opt_X_train, self.post_opt_y_train = shuffle(self.post_opt_X_train, self.post_opt_y_train)

        self.post_opt_X_train = np.array(self.post_opt_X_train)
        self.post_opt_y_train = np.array(self.post_opt_y_train)

        local_X_train = self.post_opt_X_train.copy()
        local_X_val = self.X_val.copy()
        local_X_test = self.X_test.copy()
        local_y_train = self.post_opt_y_train.copy()
        local_y_val = self.y_val.copy()
        local_y_test = self.y_test.copy()

        print(f"Best Hyperparameters: {self.filtering_options[best_hopt['filtering_type']]} | {self.tmfg_pvalues_options[best_hopt['tmfg_confidence']]}")

        model = HCNN(X_train=local_X_train,
                     X_val=local_X_val,
                     X_test=local_X_test,
                     y_train=local_y_train,
                     y_val=local_y_val,
                     y_test=local_y_test,
                     n_filters_l1=int(best_hopt['n_filters_l1']),
                     n_filters_l2=int(best_hopt['n_filters_l2']),
                     tmfg_repetitions=int(best_hopt['tmfg_iterations']),
                     tmfg_confidence=self.tmfg_pvalues_options[best_hopt['tmfg_confidence']],
                     tmfg_similarity=self.tmfg_similarities_options[best_hopt['tmfg_similarity']],
                     root_folder=self.root_folder+'out_of_sample/',
                     filtering_type=self.filtering_options[best_hopt['filtering_type']],
                     dropout_rate=self.dropout_options[best_hopt['dropout_rate']],
                     seed=self.seed,
                     )

        model.data_preparation_pipeline()
        model.fit()

        targets, preds, probs = model.predict()

        oos_time_end = time.time()
        score = classification_report(self.y_test, preds)
        print(score)
        n_parameters = (sum(p.numel() for p in model.model.parameters()))

        oos_time = oos_time_end - oos_time_start
        return model.number_of_selected_features, preds, probs, oos_time, n_parameters

    def hcnn_manager(self):
        self.root_folder = f'./Homological_FS/HCNN_Classifier_StatMatrix/Dataset_{self.dataset_id}/Seed_{self.seed}/'
        generate_folder_structure(self.root_folder)

        trial_hopt = Trials()
        best_hopt = self.hcnn_net_optimize(trial_hopt)
        best_hopt = update_keys(d=best_hopt)

        number_of_selected_features, preds, probs, oos_time, n_parameters = self.hcnn_out_of_sample_test(best_hopt=best_hopt)

        best_hopt['number_of_selected_features'] = number_of_selected_features
        best_hopt['tmfg_confidence'] = self.tmfg_pvalues_options[best_hopt['tmfg_confidence']]
        best_hopt['tmfg_similarity'] = self.tmfg_similarities_options[best_hopt['tmfg_similarity']]
        best_hopt['filtering_type'] = self.filtering_options[best_hopt['filtering_type']]
        best_hopt['dropout_rate'] = self.dropout_options[best_hopt['dropout_rate']]

        write_json_file(best_hopt, self.root_folder + 'best_hopt.csv')
        merge_probs_preds_classification(probs, preds, self.y_test, self.root_folder + 'pobs_preds.csv')

        oos_time_execution = pd.DataFrame({'time': [oos_time]})
        oos_time_execution.to_csv(self.root_folder + f'ofs_execution_time.csv', index=False)

        n_parameters = pd.DataFrame({'n_parameters': [n_parameters]})
        n_parameters.to_csv(self.root_folder + f'n_parameters.csv', index=False)

        shutil.rmtree(path=(self.root_folder + 'hyperopt/logs/'))
        shutil.rmtree(path=(self.root_folder + 'out_of_sample/logs/'))
        shutil.rmtree(path=(self.root_folder + 'hyperopt/'))
        shutil.rmtree(path=(self.root_folder + 'out_of_sample/'))
