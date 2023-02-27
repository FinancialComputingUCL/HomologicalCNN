MODEL = ['HCNN', 'RandomForest', 'XGBoost', 'CatBoost', 'LightGBM', 'TabularTransformer', 'TabNet']#, 'HRandomForest', 'HXGBoost', 'HCatBoost', 'HTabularTransformer', 'HLightGBM', 'HTabNet']
DATASET = [11, 14, 15, 16, 18, 22, 37, 40, 54, 458, 1049, 1050, 1063, 1068, 1462, 1464, 1494, 1510, 40982, 40994]

counter_1 = 1
for dataset in DATASET:
    for model in MODEL:
        file_object = open('jobs.txt', 'a')
        file_object.write(f"{counter_1:03d} {model} {dataset} \n")
        file_object.close()
        counter_1+=1