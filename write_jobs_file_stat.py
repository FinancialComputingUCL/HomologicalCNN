MODEL = ['HCNN', 'RandomForest', 'XGBoost', 'CatBoost', 'LightGBM', 'TabularTransformer', 'TabNet']
DATASET = [40994] #[11, 14, 15, 16, 18, 22, 37, 40, 54, 458, 1049, 1050, 1063, 1068, 1462, 1464, 1494, 1510, 40982, 40994]
SEED = [7521, 5163, 6576, 9706, 3679, 5506, 7570, 6741, 8895, 9267, 9435, 280, 7589, 7108, 9263, 625, 2144, 7538, 5054, 3564, 8209, 1298, 4302, 8125, 1280, 4358, 8082, 408, 8286, 8349]

counter_1 = 1
for seed in SEED:
    for dataset in DATASET:
        for model in MODEL:
            file_object = open('jobs.txt', 'a')
            file_object.write(f"{counter_1:04d} {model} {dataset} {seed}\n")
            file_object.close()
            counter_1+=1