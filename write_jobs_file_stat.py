MODEL = ['HCNN'] #['HCNN', 'RandomForest', 'XGBoost', 'CatBoost', 'LightGBM', 'TabularTransformer', 'TabNet']
DATASET = [11, 14, 15, 16, 18, 22, 37, 54, 458, 1049, 1050, 1063, 1068, 1462, 1464, 1494, 1510, 40982, 40994]
SEED = [82, 74, 21, 12, 1, 24, 84, 78, 52, 61, 555, 125, 257, 825, 290, 298, 830, 714, 159, 736, 9669, 6750, 9820, 4641, 3588, 2471, 8585, 9540, 8779, 2181]

counter_1 = 1
for seed in SEED:
    for dataset in DATASET:
        for model in MODEL:
            file_object = open('jobs.txt', 'a')
            file_object.write(f"{counter_1:04d} {model} {dataset} {seed}\n")
            file_object.close()
            counter_1 += 1