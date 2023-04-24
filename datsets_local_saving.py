from pathlib import Path

import pandas as pd

from data_management import DataManager

datasets = [11, 14, 15, 16, 18, 22, 37, 54, 458,
            1049, 1050, 1063, 1068, 1462, 1464, 1494,
            1510, 40982, 40994, 361063, 361065, 361066,
            361060, 361055, 361068, 361062, 361069, 361061,
            361275, 361277, 361273, 361270, 361278, 361276, 361274]

for dataset in datasets:
    try:
        print(f'PROCESSING DATASET {dataset}...')
        dm = DataManager(dataset, seed=0)
        data_path = f'./data/{dataset}'
        Path(data_path).mkdir(parents=True, exist_ok=True)
        X = pd.DataFrame(dm.X)
        y = pd.DataFrame(dm.y)
        X.columns = [f'Feature_{i}' for i in range(X.shape[1])]
        y.columns = ['Target']
        X.to_csv(f'{data_path}/X.csv', index=False)
        y.to_csv(f'{data_path}/y.csv', index=False)
    except:
        print(f'Dataset {dataset}: ERROR')
