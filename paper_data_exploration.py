import pandas as pd
import numpy as np
import csv

datasets = [361055, 361062, 361063, 361065, 361066, 361275, 361276, 361277, 361278]#[11, 1462, 1464, 18, 37, 15, 54, 40994, 1063, 1068, 40982, 1510, 1049, 1050, 1494, 22, 16, 458, 14]
datasets_dict_features = {}
datasets_dict_samples = {}

for dataset in datasets:
    local_df = pd.read_csv(f'./data/{dataset}/X.csv')
    print(local_df.dtypes)
    datasets_dict_features[dataset] = local_df.shape[1]
    datasets_dict_samples[dataset] = local_df.shape[0]

print(f'Average number of features: {np.mean(list(datasets_dict_features.values()))} | STD: {np.std(list(datasets_dict_features.values()))}')
print(f'Average number of samples: {np.mean(list(datasets_dict_samples.values()))} | STD: {np.std(list(datasets_dict_samples.values()))}')
print(f'Maximum number of features: {np.max(list(datasets_dict_features.values()))} | Dataset ID: {max(datasets_dict_features, key=datasets_dict_features.get)}')
print(f'Maximum number of samples: {np.max(list(datasets_dict_samples.values()))} | Dataset ID: {max(datasets_dict_samples, key=datasets_dict_samples.get)}')
print(f'Minimum number of features: {np.min(list(datasets_dict_features.values()))} | Dataset ID: {min(datasets_dict_features, key=datasets_dict_features.get)}')
print(f'Minimum number of samples: {np.min(list(datasets_dict_samples.values()))} | Dataset ID: {min(datasets_dict_samples, key=datasets_dict_samples.get)}')

number_of_classes = {}

for dataset in datasets:
    local_df = pd.read_csv(f'./data/{dataset}/y.csv')
    number_of_classes[dataset] = len(local_df['Target'].unique())

print(number_of_classes)


def calculate_class_proportions(csv_file):
    class_counts = {}

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if present

        total_samples = 0

        for row in reader:
            if len(row) > 0:
                label = row[0]

                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1

                total_samples += 1

    class_proportions = {int(label): count for label, count in class_counts.items()} #/ total_samples
    return class_proportions


for dataset in datasets:
    print(f"DATASET: {dataset}")

    samples = pd.read_csv(f'./data/{dataset}/X.csv')
    print(f'# features: {samples.shape[1]}')
    print(f'# samples: {samples.shape[0]}')

    csv_file_path = f'./data/{dataset}/y.csv'
    proportions = calculate_class_proportions(csv_file_path)

    print(f"Class Proportions:")
    print(dict(sorted(proportions.items())))

    print('----------------------------------------')