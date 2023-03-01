import os
import json
import time

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


def generate_folder_structure(folder_structure):
    if not os.path.exists(folder_structure):
        os.makedirs(folder_structure)


def write_json_file(data, filename):
    pd.DataFrame(data, index=[0]).to_csv(filename, index=False)


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time of function '{func.__name__}': {end - start:.6f} seconds")
        result['execution_time'] = f'{end - start:.6f}'
        return result
    return wrapper


def convert_to_string(array):
    return np.vectorize(lambda x: f"Class_{x}")(array)


def merge_probs_preds_classification(array1, array2, targets, filename):
    #classes = convert_to_string(np.arange(0, array1.shape[1]))
    #df1 = pd.DataFrame(array1, columns=classes)
    df2 = pd.DataFrame(array2, columns=['Pred'])
    df3 = pd.DataFrame(targets, columns=['Target'])
    #df = pd.concat([df1, df2, df3], axis=1)
    df = pd.concat([df2, df3], axis=1)
    df.to_csv(filename, index=False)


def update_keys(d):
    keys_to_delete = []
    for key in d:
        new_key = key + "_"
        if new_key in d:
            d[key] = d[new_key]
            keys_to_delete.append(new_key)
    for key in keys_to_delete:
        del d[key]
    return d


def scaling(X_train, X_additional, choice):
    if choice == 'Standard_Scaling':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_additional = scaler.transform(X_additional)
    elif choice == 'MinMax_Scaling':
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_additional = scaler.transform(X_additional)
    elif choice == 'MaxAbs_Scaling':
        scaler = MaxAbsScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_additional = scaler.transform(X_additional)
    elif choice == 'Robust_Scaling':
        scaler = RobustScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_additional = scaler.transform(X_additional)
    else:
        X_train = np.array(X_train)
        X_additional = np.array(X_additional)

    return X_train, X_additional