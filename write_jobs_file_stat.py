MODEL = ['HCNN'] #'TabNet', 'TabularTransformer',  #['RandomForest', 'LogisticRegression', 'XGBoost', 'CatBoost']
DATASET = [11, 14, 15, 16, 18, 22, 37, 54, 458, 1049, 1050, 1063, 1068, 1462, 1464, 1494, 1510, 40982, 40994]
SEED = [9433, 67822, 9822127, 8279, 12555, 903, 12, 7687, 22443, 190] #[87, 12, 33, 113, 339, 543, 3226, 7687, 12555, 827, 1666, 22443, 999, 16373, 9433, 594312, 7, 444, 67822, 5588, 989, 5410, 190, 8279, 5550, 5950, 903, 98721, 56632, 9822127]

counter_1 = 1
for model in MODEL:
    for seed in SEED:
        for dataset in DATASET:
            file_object = open('repairing_jobs.txt', 'a')
            file_object.write(f"{counter_1:04d} {model} {dataset} {seed}\n")
            file_object.close()
            counter_1 += 1