from HCNN import *
from data_management import *

from sklearn.metrics import classification_report

if __name__ == '__main__':
    dm = DataManager(dataset_id=22, seed=234)
    X_train, X_val, X_test, y_train, y_val, y_test = dm.get_data()

    my_HCNN = HCNN(X_train = X_train,
                   X_val = X_val,
                   X_test = X_test,
                   y_train = y_train,
                   y_val = y_val,
                   y_test = y_test,
                   n_filters_l1 = 4,
                   n_filters_l2 = 32,
                   tmfg_repetitions = 1000,
                   tmfg_confidence = 90,
                   tmfg_similarity = 'spearman',
                   learning_rate = 0.0001,
                   max_epochs=1500,
                   T=1)

    X_train, X_val, X_test, y_train, y_val, y_test, net = my_HCNN.data_preparation_pipeline()
    net.fit(X_train, y_train)
    preds = net.predict(X_test)
    print(classification_report(y_test, preds))
