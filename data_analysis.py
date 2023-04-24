import pandas as pd
import numpy as np

from sklearn.metrics import *
from scipy.stats import ttest_rel
from sklearn.preprocessing import LabelBinarizer

import params

def local_roc_auc_score(y_test, y_prob, average="macro"):
    n_classes = np.unique(y_test)

    if len(n_classes) > 2:
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovo", average=average)
    else:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
        roc_auc_ovr = auc(fpr, tpr)

    return roc_auc_ovr


if __name__ == '__main__':
    seeds = [82, 74, 21, 12, 1, 24, 84, 78, 52, 61, 555, 125, 257, 825, 290, 298, 830, 714, 159, 736, 9669, 6750, 9820, 4641, 3588, 2471, 8585, 9540, 8779, 2181]
    models = ['RandomForestClassifier',
              'XGBoostClassifier',
              'CatBoostClassifier',
              'LightGBM_Classifier',
              'TabNet_Classifier',
              'TabPFN_Classifier',
              'HCNN_Classifier',
              ]
    datasets = [11, 14, 15, 16, 18, 22, 37, 54, 458, 1049, 1050, 1063, 1068, 1462, 1464, 1494, 1510, 40982, 40994]

    final_analysis_df = {}
    statistically_different = 0
    not_statistically_different = 0
    statistically_different_list = []
    not_statistically_different_list = []

    wilcoxon_test = {}

    for dataset in datasets:
        print(f'DATASET ID: {dataset}')
        performances_list = {}
        for model in models:
            metrics_list = []
            params_dict = []
            for seed in seeds:
                try:
                    probs_preds = pd.read_csv(f'./Results/{model}/Dataset_{dataset}/Seed_{seed}/pobs_preds.csv')
                    best_hopt = pd.read_csv(f'./Results/{model}/Dataset_{dataset}/Seed_{seed}/best_hopt.csv')

                    target = probs_preds['Target']
                    pred = probs_preds['Pred']
                    execution_time = best_hopt['execution_time'].values[0]
                    probs = probs_preds.iloc[:, :-2]

                    #metrics_list.append(f1_score(target, pred, average='macro'))
                    metrics_list.append(accuracy_score(target, pred))
                    #metrics_list.append(local_roc_auc_score(np.array(target), np.array(probs), average='macro'))
                    params_dict.append(execution_time)

                except:
                    print(f'{model} - Dataset_{dataset} - Seed_{seed}: ERROR')

            try:
                print(f'{model} - Dataset_{dataset}: {np.mean(metrics_list):.2f} +/- {np.std(metrics_list):.2f} | Execution Time: {np.mean(params_dict):.2f}')
                final_analysis_df[f'{model}'] = metrics_list
                performances_list[f'{model}'] = np.mean(metrics_list)
            except:
                print(f'{model} - Dataset_{dataset}: ERROR')

        analysis_matrix = np.zeros((len(models), len(models)))
        for m in models:
            for n in models:
                if m != n:
                    try:
                        t_stat, p_val = ttest_rel(final_analysis_df[m], final_analysis_df[n])
                        if p_val < 0.001 and performances_list[m] > performances_list[n]:
                            analysis_matrix[models.index(m)][models.index(n)] = '1' # UPPER
                        elif p_val < 0.001 and performances_list[m] < performances_list[n]:
                            analysis_matrix[models.index(m)][models.index(n)] = '2' # LOWER
                        elif p_val > 0.001:
                            analysis_matrix[models.index(m)][models.index(n)] = '3' # EQUAL
                    except:
                        pass
        analysis_matrix = pd.DataFrame(analysis_matrix)
        analysis_matrix.columns = models
        analysis_matrix.index = models
        print(analysis_matrix)

        wilcoxon_test[f'Dateset_{dataset}'] = performances_list

        print('----------------------------------------')

    df = pd.DataFrame(wilcoxon_test)
    df = df.reset_index().melt(id_vars='index', var_name='Dataset', value_name='Score')
    df = df.rename(columns={'index': 'Classifier'})
    df.columns = ['classifier_name', 'dataset_name', 'accuracy']

    from wilcoxon_test import *

    draw_cd_diagram(df_perf=df, title='Accuracy', labels=True)