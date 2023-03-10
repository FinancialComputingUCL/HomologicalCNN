import pandas as pd
import numpy as np

from sklearn.metrics import *
from scipy.stats import ttest_rel

import params

if __name__ == '__main__':
    seeds = [7521, 5163, 6576, 9706, 3679, 5506, 7570, 6741, 8895, 9267, 9435, 280, 7589, 7108, 9263, 625, 2144, 7538, 5054, 3564, 8209, 1298, 4302, 8125, 1280, 4358, 8082, 408, 8286, 8349]
    models = ['RandomForestClassifier',
              'XGBoostClassifier',
              'CatBoostClassifier',
              'LightGBM_Classifier',
              'TabNet_Classifier',
              'TabPFN_Classifier',
              'HCNN_Classifier',
              ]
    datasets = [11, 14, 15, 16, 18, 22, 37, 40, 54, 458, 1049, 1050, 1063, 1068, 1462, 1464, 1494, 1510, 40982, 40994]

    final_analysis_df = {}
    statistically_different = 0
    not_statistically_different = 0
    statistically_different_list = []
    not_statistically_different_list = []

    for dataset in datasets:
        print(f'DATASET ID: {dataset}')
        for model in models:
            metrics_list = []
            params_dict = []
            for seed in seeds:
                try:
                    probs_preds = pd.read_csv(f'./Results/Exp_3/{model}/Dataset_{dataset}/Seed_{seed}/pobs_preds.csv')
                    best_hopt = pd.read_csv(f'./Results/Exp_3/{model}/Dataset_{dataset}/Seed_{seed}/best_hopt.csv')

                    target = probs_preds['Target']
                    pred = probs_preds['Pred']
                    execution_time = best_hopt['execution_time'].values[0]

                    metrics_list.append(accuracy_score(target, pred))
                    params_dict.append(execution_time)
                except:
                    print(f'{model} - Dataset_{dataset} - Seed_{seed}: ERROR')

            try:
                print(f'{model} - Dataset_{dataset}: {np.mean(metrics_list):.2f} +/- {np.std(metrics_list):.2f} | Execution Time: {np.mean(params_dict):.2f}')
                final_analysis_df[f'{model}'] = metrics_list
            except:
                print(f'{model} - Dataset_{dataset}: ERROR')

        # Calculate the t-value and p-value
        try:
            t_stat, p_val = ttest_rel(final_analysis_df['TabPFN_Classifier'], final_analysis_df['HCNN_Classifier'])

            if p_val < 0.001:
                print("The two classifiers are statistically different.")
                statistically_different += 1
                statistically_different_list.append(dataset)
            else:
                print("The two classifiers are not statistically different.")
                not_statistically_different += 1
                not_statistically_different_list.append(dataset)
        except:
            pass

        print('----------------------------------------')

    print(f'Statistically different: {statistically_different} | List: {statistically_different_list}')
    print(f'Not statistically different: {not_statistically_different} | List: {not_statistically_different_list}')
