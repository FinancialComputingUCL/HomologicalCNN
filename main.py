import argparse

from models_management import *

parser = argparse.ArgumentParser(description='HCNN Experiments.')
parser.add_argument('--model', type=str, default='HCNN', help="Model to be run.")
parser.add_argument('--dataset_id', type=int, default=361277, help="Dataset to be considered.")
parser.add_argument('--seed', type=int, default=9822127, help="Seed to be used.")
args = parser.parse_args()

if __name__ == '__main__':

    seed = args.seed
    dm = DataManager(dataset_id=args.dataset_id, seed=seed)
    X_train, X_val, X_test, y_train, y_val, y_test = dm.get_data()

    if args.model == 'RandomForest':
        mm = ModelsManager(model='RandomForest', seed=seed, dataset_id=args.dataset_id,
                           X_train=X_train, X_val=X_val, X_test=X_test,
                           y_train=y_train, y_val=y_val, y_test=y_test)
        mm.random_forest_manager()

    elif args.model == 'LogisticRegression':
        mm = ModelsManager(model='LogisticRegression', seed=seed, dataset_id=args.dataset_id,
                           X_train=X_train, X_val=X_val, X_test=X_test,
                           y_train=y_train, y_val=y_val, y_test=y_test)
        mm.logistic_regression_manager()

    elif args.model == 'XGBoost':
        mm = ModelsManager(model='XGBoost', seed=seed, dataset_id=args.dataset_id,
                           X_train=X_train, X_val=X_val, X_test=X_test,
                           y_train=y_train, y_val=y_val, y_test=y_test)
        mm.xgboost_manager()

    elif args.model == 'CatBoost':
        mm = ModelsManager(model='CatBoost', seed=seed, dataset_id=args.dataset_id,
                           X_train=X_train, X_val=X_val, X_test=X_test,
                           y_train=y_train, y_val=y_val, y_test=y_test)
        mm.cat_boost_manager()

    elif args.model == 'LightGBM':
        mm = ModelsManager(model='LightGBM', seed=seed, dataset_id=args.dataset_id,
                           X_train=X_train, X_val=X_val, X_test=X_test,
                           y_train=y_train, y_val=y_val, y_test=y_test)
        mm.lightgbm_manager()

    elif args.model == 'MLP':
        mm = ModelsManager(model='MLP', seed=seed, dataset_id=args.dataset_id,
                           X_train=X_train, X_val=X_val, X_test=X_test,
                           y_train=y_train, y_val=y_val, y_test=y_test)
        mm.mlp_manager()

    elif args.model == 'TabularTransformer':
        mm = ModelsManager(model='TabularTransformer', seed=seed, dataset_id=args.dataset_id,
                           X_train=X_train, X_val=X_val, X_test=X_test,
                           y_train=y_train, y_val=y_val, y_test=y_test)
        mm.tab_pfn_manager()

    elif args.model == 'TabNet':
        mm = ModelsManager(model='TabNet', seed=seed, dataset_id=args.dataset_id,
                           X_train=X_train, X_val=X_val, X_test=X_test,
                           y_train=y_train, y_val=y_val, y_test=y_test)
        mm.tab_net_manager()

    elif args.model == 'HCNN':
        mm = ModelsManager(model='HCNN', seed=seed, dataset_id=args.dataset_id,
                           X_train=X_train, X_val=X_val, X_test=X_test,
                           y_train=y_train, y_val=y_val, y_test=y_test)
        mm.hcnn_manager()