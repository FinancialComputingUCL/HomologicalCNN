import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer


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
    datasets = [11, 1462, 1464, 18, 37, 15, 54, 40994, 1063, 1068, 40982, 1510, 1049, 1050, 1494, 22, 16, 458, 14]#[11, 14, 15, 16, 18, 22, 37, 54, 458, 1049, 1050, 1063, 1068, 1462, 1464, 1494, 1510, 40982, 40994]
    regularization_factor = 0.05

    final_analysis_df = {}
    statistically_different = 0
    not_statistically_different = 0
    statistically_different_list = []
    not_statistically_different_list = []

    output_dict = {}
    output_list = []

    wilcoxon_test = {}

    running_time = {'RandomForestClassifier': [], 'XGBoostClassifier': [], 'CatBoostClassifier': [], 'LightGBM_Classifier': [], 'TabNet_Classifier': [], 'TabPFN_Classifier': [], 'HCNN_Classifier': []}

    text = ''
    for dataset in datasets:
        print(f'DATASET ID: {dataset}')
        performances_list = {}
        text += '\n'
        for model in models:
            metrics_list = []
            params_dict = []
            for seed in seeds:
                try:
                    if model != 'HCNN_Classifier':
                        probs_preds = pd.read_csv(f'./Results/{model}/Dataset_{dataset}/Seed_{seed}/pobs_preds.csv')
                        best_hopt = pd.read_csv(f'./Results/{model}/Dataset_{dataset}/Seed_{seed}/best_hopt.csv')
                    else:
                        probs_preds = pd.read_csv(f'./Results/{model}/Regularization_{regularization_factor}/Dataset_{dataset}/Seed_{seed}/pobs_preds.csv')
                        best_hopt = pd.read_csv(f'./Results/{model}/Regularization_{regularization_factor}/Dataset_{dataset}/Seed_{seed}/best_hopt.csv')

                    target = probs_preds['Target']
                    pred = probs_preds['Pred']
                    execution_time = best_hopt['execution_time'].values[0]
                    probs = probs_preds.iloc[:, :-2]

                    metrics_list.append(matthews_corrcoef(target, pred))
                    #metrics_list.append(f1_score(target, pred, average='macro'))
                    #metrics_list.append(accuracy_score(target, pred))
                    #metrics_list.append(local_roc_auc_score(np.array(target), np.array(probs), average='macro'))
                    params_dict.append(execution_time)

                except:
                    print(f'{model} - Dataset_{dataset} - Seed_{seed}: ERROR')

            try:
                print(f'{model} - Dataset_{dataset}: {np.mean(metrics_list):.2f} +/- {np.std(metrics_list):.2f} | Execution Time: {np.mean(params_dict):.2f}')
                running_time[f'{model}'].append(np.mean(params_dict))
                text += f'{np.mean(metrics_list):.2f}+/-{np.std(metrics_list):.2f}\t'
                final_analysis_df[f'{model}'] = metrics_list
                performances_list[f'{model}'] = np.mean(metrics_list)
            except:
                print(f'{model} - Dataset_{dataset}: ERROR')

        analysis_matrix = np.zeros((len(models), len(models)))
        for m in models:
            for n in models:
                if m != n:
                    try:
                        #print(final_analysis_df[m])
                        #print(final_analysis_df[n])
                        first = ['%.2f' % elem for elem in final_analysis_df[m]]
                        second = ['%.2f' % elem for elem in final_analysis_df[n]]
                        first = [float(i) for i in first]
                        second = [float(i) for i in second]
                        #print(first)
                        #print(second)
                        #print('-------------------')
                        t_stat, p_val = ttest_rel(first, second)
                        if p_val < 0.01 and performances_list[m] > performances_list[n]:
                            analysis_matrix[models.index(m)][models.index(n)] = '1' # UPPER
                        elif p_val < 0.01 and performances_list[m] < performances_list[n]:
                            analysis_matrix[models.index(m)][models.index(n)] = '2' # LOWER
                        elif p_val > 0.01:
                            analysis_matrix[models.index(m)][models.index(n)] = '3' # EQUAL
                    except:
                        pass
        analysis_matrix = pd.DataFrame(analysis_matrix)
        analysis_matrix.columns = models
        analysis_matrix.index = models
        print(analysis_matrix)
        analysis_matrix.to_csv(f'./matrices/{dataset}.csv')

        wilcoxon_test[f'Dateset_{dataset}'] = performances_list

        print('----------------------------------------')

    df = pd.DataFrame(wilcoxon_test)
    df = df.reset_index().melt(id_vars='index', var_name='Dataset', value_name='Score')
    df = df.rename(columns={'index': 'Classifier'})
    df.columns = ['classifier_name', 'dataset_name', 'accuracy']

    plt.rcParams["figure.figsize"] = (18, 10)
    plt.rcParams["font.size"] = 14
    x_ax = [f'{str(x)}' for x in datasets]
    for k in running_time.keys():
        if k != 'HCNN_Classifier':
            plt.plot(x_ax, running_time[k], label=k, alpha=0.2, linewidth=3)
        else:
            plt.plot(x_ax, running_time[k], label=k, linewidth=3)

    plt.xlabel('Dataset')
    plt.ylabel('Average Running Time (s)')
    plt.legend()
    plt.savefig('running_time.pdf', dpi=500)