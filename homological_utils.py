import numpy as np

from tmfg_bootstrapped import *
from tmfg_core import *

from sklearn.preprocessing import RobustScaler

from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader


def get_final_X_4(X, final_b_cliques_4):
    final_X = None

    for e, c in enumerate(final_b_cliques_4):
        if final_X is None:
            final_X = X[final_b_cliques_4[e]]
        else:
            final_X = pd.concat(
                [final_X, X[final_b_cliques_4[e]]], ignore_index=True, axis=1
            )
    return final_X


def get_final_X_3(X, final_b_cliques_3):
    final_X = None

    for e, c in enumerate(final_b_cliques_3):
        if final_X is None:
            final_X = X[final_b_cliques_3[e]]
        else:
            final_X = pd.concat(
                [final_X, X[final_b_cliques_3[e]]], ignore_index=True, axis=1
            )
    return final_X


def get_final_X_2(X, final_b_cliques_2):
    final_X = None

    for e, c in enumerate(final_b_cliques_2):
        if final_X is None:
            final_X = X[final_b_cliques_2[e]]
        else:
            final_X = pd.concat(
                [final_X, X[final_b_cliques_2[e]]], ignore_index=True, axis=1
            )
    return final_X


def h_input_transform(X_train, X_val, X_test, y_train, y_val, y_test, tmfg_repetitions, tmfg_confidence,
                      tmfg_similarity):
    cliques, separators, original_tmfg, _, adjacency_matrix = TMFG_Bootstrapped(X_train,
                                                                                tmfg_similarity,
                                                                                tmfg_repetitions,
                                                                                tmfg_confidence,
                                                                                parallel=True).compute_tmfg_bootstrapping()

    c = nx.degree_centrality(adjacency_matrix)

    keys = np.array(list(c.keys()))
    values = np.array(list(c.values()))
    nodes_list = sorted(list(keys[values != 0]))

    simplexes = []

    x = None
    x_train = None
    x_val = None
    x_test = None

    for i in nx.enumerate_all_cliques(original_tmfg):
        if len(i) == 2:
            simplexes.append(sorted(i))

    b_cliques_4 = []
    b_cliques_3 = []
    b_cliques_2 = []

    b_cliques_all = nx.enumerate_all_cliques(adjacency_matrix)

    for i in b_cliques_all:
        if len(i) == 2:
            b_cliques_2.append(sorted(i))
        if len(i) == 3:
            b_cliques_3.append(sorted(i))
        if len(i) == 4:
            b_cliques_4.append(sorted(i))

    final_b_cliques_4 = []

    for c in cliques:
        if sorted(c) in b_cliques_4:
            final_b_cliques_4.append(sorted(c))

    final_b_cliques_3 = []

    for c in separators:
        if sorted(c) in b_cliques_3:
            final_b_cliques_3.append(sorted(c))

    final_b_cliques_2 = []

    for c in simplexes:
        if sorted(c) in b_cliques_2:
            final_b_cliques_2.append(sorted(c))

    new_b_cliques_3 = []

    if len(final_b_cliques_4) == 0:
        new_b_cliques_3 = final_b_cliques_3

    else:
        for t in final_b_cliques_3:
            flag = False
            for f in final_b_cliques_4:
                if set(t).issubset(set(f)):
                    flag = True
            if not flag:
                new_b_cliques_3.append(t)

    final_b_cliques_3 = new_b_cliques_3

    new_b_cliques_2 = []

    if len(final_b_cliques_3) == 0:
        new_b_cliques_2 = final_b_cliques_2

    else:
        for t in final_b_cliques_2:
            flag = False
            for f in final_b_cliques_3:
                if set(t).issubset(set(f)):
                    flag = True
            if not flag:
                new_b_cliques_2.append(t)

    final_b_cliques_2 = new_b_cliques_2

    new_b_cliques_2 = []

    if len(final_b_cliques_4) == 0:
        new_b_cliques_2 = final_b_cliques_2

    else:
        for t in final_b_cliques_2:
            flag = False
            for f in final_b_cliques_4:
                if set(t).issubset(set(f)):
                    flag = True
            if not flag:
                new_b_cliques_2.append(t)

    final_b_cliques_2 = new_b_cliques_2

    try:
        if x is None:
            final_train_X_4 = get_final_X_4(X_train, final_b_cliques_4)
            final_val_X_4 = get_final_X_4(X_val, final_b_cliques_4)
            final_test_X_4 = get_final_X_4(X_test, final_b_cliques_4)
        else:
            final_train_X_4 = get_final_X_4(x_train, final_b_cliques_4)
            final_val_X_4 = get_final_X_4(x_val, final_b_cliques_4)
            final_test_X_4 = get_final_X_4(x_test, final_b_cliques_4)
    except:
        final_train_X_4 = None
        final_val_X_4 = None
        final_test_X_4 = None

    try:
        if x is None:
            final_train_X_3 = get_final_X_3(X_train, final_b_cliques_3)
            final_val_X_3 = get_final_X_3(X_val, final_b_cliques_3)
            final_test_X_3 = get_final_X_3(X_test, final_b_cliques_3)
        else:
            final_train_X_3 = get_final_X_3(x_train, final_b_cliques_3)
            final_val_X_3 = get_final_X_3(x_val, final_b_cliques_3)
            final_test_X_3 = get_final_X_3(x_test, final_b_cliques_3)
    except:
        final_train_X_3 = None
        final_val_X_3 = None
        final_test_X_3 = None

    try:
        if x is None:
            final_train_X_2 = get_final_X_2(X_train, final_b_cliques_2)
            final_val_X_2 = get_final_X_2(X_val, final_b_cliques_2)
            final_test_X_2 = get_final_X_2(X_test, final_b_cliques_2)
        else:
            final_train_X_2 = get_final_X_2(x_train, final_b_cliques_2)
            final_val_X_2 = get_final_X_2(x_val, final_b_cliques_2)
            final_test_X_2 = get_final_X_2(x_test, final_b_cliques_2)
    except:
        final_train_X_2 = None
        final_val_X_2 = None
        final_test_X_2 = None

    try:
        scaler = RobustScaler()

        final_train_X_4 = scaler.fit_transform(final_train_X_4)
        final_val_X_4 = scaler.transform(final_val_X_4)
        final_test_X_4 = scaler.transform(final_test_X_4)
    except:
        pass

    try:
        scaler = RobustScaler()

        final_train_X_3 = scaler.fit_transform(final_train_X_3)
        final_val_X_3 = scaler.transform(final_val_X_3)
        final_test_X_3 = scaler.transform(final_test_X_3)
    except:
        pass

    try:
        scaler = RobustScaler()

        final_train_X_2 = scaler.fit_transform(final_train_X_2)
        final_val_X_2 = scaler.transform(final_val_X_2)
        final_test_X_2 = scaler.transform(final_test_X_2)
    except:
        pass

    try:
        final_train_X_4 = final_train_X_4.reshape(final_train_X_4.shape[0], 1, final_train_X_4.shape[1], 1)
        final_val_X_4 = final_val_X_4.reshape(final_val_X_4.shape[0], 1, final_val_X_4.shape[1], 1)
        final_test_X_4 = final_test_X_4.reshape(final_test_X_4.shape[0], 1, final_test_X_4.shape[1], 1)
    except:
        pass

    try:
        final_train_X_3 = final_train_X_3.reshape(final_train_X_3.shape[0], 1, final_train_X_3.shape[1], 1)
        final_val_X_3 = final_val_X_3.reshape(final_val_X_3.shape[0], 1, final_val_X_3.shape[1], 1)
        final_test_X_3 = final_test_X_3.reshape(final_test_X_3.shape[0], 1, final_test_X_3.shape[1], 1)
    except:
        pass

    try:
        final_train_X_2 = final_train_X_2.reshape(final_train_X_2.shape[0], 1, final_train_X_2.shape[1], 1)
        final_val_X_2 = final_val_X_2.reshape(final_val_X_2.shape[0], 1, final_val_X_2.shape[1], 1)
        final_test_X_2 = final_test_X_2.reshape(final_test_X_2.shape[0], 1, final_test_X_2.shape[1], 1)
    except:
        pass

    print(f'# Cliques: {len(final_b_cliques_4)}')
    print(f'# Triangles: {len(final_b_cliques_3)}')
    print(f'# Simplexes: {len(final_b_cliques_2)}')

    shape_4 = None
    shape_3 = None
    shape_2 = None

    try:
        shape_4 = final_train_X_4.shape[2]
    except:
        pass

    try:
        shape_3 = final_train_X_3.shape[2]
    except:
        pass

    try:
        shape_2 = final_train_X_2.shape[2]
    except:
        pass

    try:
        final_train_X_4 = final_train_X_4.reshape(final_train_X_4.shape[0], final_train_X_4.shape[2])
    except:
        pass

    try:
        final_train_X_3 = final_train_X_3.reshape(final_train_X_3.shape[0], final_train_X_3.shape[2])
    except:
        pass

    try:
        final_train_X_2 = final_train_X_2.reshape(final_train_X_2.shape[0], final_train_X_2.shape[2])
    except:
        pass

    try:
        final_val_X_4 = final_val_X_4.reshape(final_val_X_4.shape[0], final_val_X_4.shape[2])
    except:
        pass

    try:
        final_val_X_3 = final_val_X_3.reshape(final_val_X_3.shape[0], final_val_X_3.shape[2])
    except:
        pass

    try:
        final_val_X_2 = final_val_X_2.reshape(final_val_X_2.shape[0], final_val_X_2.shape[2])
    except:
        pass

    try:
        final_test_X_4 = final_test_X_4.reshape(final_test_X_4.shape[0], final_test_X_4.shape[2])
    except:
        pass

    try:
        final_test_X_3 = final_test_X_3.reshape(final_test_X_3.shape[0], final_test_X_3.shape[2])
    except:
        pass

    try:
        final_test_X_2 = final_test_X_2.reshape(final_test_X_2.shape[0], final_test_X_2.shape[2])
    except:
        pass

    if final_train_X_4 is not None and final_train_X_3 is not None and final_train_X_2 is not None:
        X_train = {'tetrahedra': final_train_X_4, 'triangles': final_train_X_3, 'simplex': final_train_X_2}
        X_val = {'tetrahedra': final_val_X_4, 'triangles': final_val_X_3, 'simplex': final_val_X_2}
        X_test = {'tetrahedra': final_test_X_4, 'triangles': final_test_X_3, 'simplex': final_test_X_2}
    if final_train_X_4 is None and final_train_X_3 is not None and final_train_X_2 is not None:
        X_train = {'triangles': final_train_X_3, 'simplex': final_train_X_2}
        X_val = {'triangles': final_val_X_3, 'simplex': final_val_X_2}
        X_test = {'triangles': final_test_X_3, 'simplex': final_test_X_2}
    if final_train_X_4 is not None and final_train_X_3 is None and final_train_X_2 is not None:
        X_train = {'tetrahedra': final_train_X_4, 'simplex': final_train_X_2}
        X_val = {'tetrahedra': final_val_X_4, 'simplex': final_val_X_2}
        X_test = {'tetrahedra': final_test_X_4, 'simplex': final_test_X_2}
    if final_train_X_4 is not None and final_train_X_3 is None and final_train_X_2 is None:
        X_train = {'tetrahedra': final_train_X_4}
        X_val = {'tetrahedra': final_val_X_4}
        X_test = {'tetrahedra': final_test_X_4}
    if final_train_X_4 is None and final_train_X_3 is not None and final_train_X_2 is None:
        X_train = {'triangles': final_train_X_3}
        X_val = {'triangles': final_val_X_3}
        X_test = {'triangles': final_test_X_3}
    if final_train_X_4 is None and final_train_X_3 is None and final_train_X_2 is not None:
        X_train = {'simplex': final_train_X_2}
        X_val = {'simplex': final_val_X_2}
        X_test = {'simplex': final_test_X_2}

    return len(nodes_list), shape_4, shape_3, shape_2, X_train, X_val, X_test, y_train, y_val, y_test


def prepare_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    combined_loaders_train = None
    combined_loaders_val = None
    combined_loaders_test = None

    # Dataloaders training side.
    if 'tetrahedra' in X_train.keys() and 'triangles' in X_train.keys() and 'simplex' in X_train.keys():
        loader_tetrahedra = DataLoader(X_train['tetrahedra'], batch_size=batch_size, drop_last=True)
        loader_triangles = DataLoader(X_train['triangles'], batch_size=batch_size, drop_last=True)
        loader_simplex = DataLoader(X_train['simplex'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_train, batch_size=batch_size, drop_last=True)

        loaders = {"tetrahedra": loader_tetrahedra, "triangles": loader_triangles, 'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_train = CombinedLoader(loaders)

    if 'tetrahedra' in X_train.keys() and 'triangles' in X_train.keys() and 'simplex' not in X_train.keys():
        loader_tetrahedra = DataLoader(X_train['tetrahedra'], batch_size=batch_size, drop_last=True)
        loader_triangles = DataLoader(X_train['triangles'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_train, batch_size=batch_size, drop_last=True)

        loaders = {"tetrahedra": loader_tetrahedra, "triangles": loader_triangles, 'targets': loader_targets}
        combined_loaders_train = CombinedLoader(loaders)

    if 'tetrahedra' in X_train.keys() and 'triangles' not in X_train.keys() and 'simplex' in X_train.keys():
        loader_tetrahedra = DataLoader(X_train['tetrahedra'], batch_size=batch_size, drop_last=True)
        loader_simplex = DataLoader(X_train['simplex'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_train, batch_size=batch_size, drop_last=True)

        loaders = {"tetrahedra": loader_tetrahedra, 'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_train = CombinedLoader(loaders)

    if 'tetrahedra' in X_train.keys() and 'triangles' not in X_train.keys() and 'simplex' not in X_train.keys():
        loader_tetrahedra = DataLoader(X_train['tetrahedra'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_train, batch_size=batch_size, drop_last=True)

        loaders = {"tetrahedra": loader_tetrahedra, 'targets': loader_targets}
        combined_loaders_train = CombinedLoader(loaders)

    if 'tetrahedra' not in X_train.keys() and 'triangles' in X_train.keys() and 'simplex' in X_train.keys():
        loader_triangles = DataLoader(X_train['triangles'], batch_size=batch_size, drop_last=True)
        loader_simplex = DataLoader(X_train['simplex'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_train, batch_size=batch_size, drop_last=True)

        loaders = {"triangles": loader_triangles, 'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_train = CombinedLoader(loaders)

    if 'tetrahedra' not in X_train.keys() and 'triangles' in X_train.keys() and 'simplex' not in X_train.keys():
        loader_triangles = DataLoader(X_train['triangles'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_train, batch_size=batch_size, drop_last=True)

        loaders = {"triangles": loader_triangles, 'targets': loader_targets}
        combined_loaders_train = CombinedLoader(loaders)

    if 'tetrahedra' not in X_train.keys() and 'triangles' not in X_train.keys() and 'simplex' in X_train.keys():
        loader_simplex = DataLoader(X_train['simplex'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_train, batch_size=batch_size, drop_last=True)

        loaders = {'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_train = CombinedLoader(loaders)

    # Dataloaders validation side.
    if 'tetrahedra' in X_val.keys() and 'triangles' in X_val.keys() and 'simplex' in X_val.keys():
        loader_tetrahedra = DataLoader(X_val['tetrahedra'], batch_size=batch_size, drop_last=True)
        loader_triangles = DataLoader(X_val['triangles'], batch_size=batch_size, drop_last=True)
        loader_simplex = DataLoader(X_val['simplex'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_val, batch_size=batch_size, drop_last=True)

        loaders = {"tetrahedra": loader_tetrahedra, "triangles": loader_triangles, 'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_val = CombinedLoader(loaders)

    if 'tetrahedra' in X_val.keys() and 'triangles' in X_val.keys() and 'simplex' not in X_val.keys():
        loader_tetrahedra = DataLoader(X_val['tetrahedra'], batch_size=batch_size, drop_last=True)
        loader_triangles = DataLoader(X_val['triangles'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_val, batch_size=batch_size, drop_last=True)

        loaders = {"tetrahedra": loader_tetrahedra, "triangles": loader_triangles, 'targets': loader_targets}
        combined_loaders_val = CombinedLoader(loaders)

    if 'tetrahedra' in X_val.keys() and 'triangles' not in X_val.keys() and 'simplex' in X_val.keys():
        loader_tetrahedra = DataLoader(X_val['tetrahedra'], batch_size=batch_size, drop_last=True)
        loader_simplex = DataLoader(X_val['simplex'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_val, batch_size=batch_size, drop_last=True)

        loaders = {"tetrahedra": loader_tetrahedra, 'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_val = CombinedLoader(loaders)

    if 'tetrahedra' in X_val.keys() and 'triangles' not in X_val.keys() and 'simplex' not in X_val.keys():
        loader_tetrahedra = DataLoader(X_val['tetrahedra'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_val, batch_size=batch_size, drop_last=True)

        loaders = {"tetrahedra": loader_tetrahedra, 'targets': loader_targets}
        combined_loaders_val = CombinedLoader(loaders)

    if 'tetrahedra' not in X_val.keys() and 'triangles' in X_val.keys() and 'simplex' in X_val.keys():
        loader_triangles = DataLoader(X_val['triangles'], batch_size=batch_size, drop_last=True)
        loader_simplex = DataLoader(X_val['simplex'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_val, batch_size=batch_size, drop_last=True)

        loaders = {"triangles": loader_triangles, 'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_val = CombinedLoader(loaders)

    if 'tetrahedra' not in X_val.keys() and 'triangles' in X_val.keys() and 'simplex' not in X_val.keys():
        loader_triangles = DataLoader(X_val['triangles'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_val, batch_size=batch_size, drop_last=True)

        loaders = {"triangles": loader_triangles, 'targets': loader_targets}
        combined_loaders_val = CombinedLoader(loaders)

    if 'tetrahedra' not in X_val.keys() and 'triangles' not in X_val.keys() and 'simplex' in X_val.keys():
        loader_simplex = DataLoader(X_val['simplex'], batch_size=batch_size, drop_last=True)
        loader_targets = DataLoader(y_val, batch_size=batch_size, drop_last=True)

        loaders = {'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_val = CombinedLoader(loaders)

    # Dataloaders test side.
    if 'tetrahedra' in X_test.keys() and 'triangles' in X_test.keys() and 'simplex' in X_test.keys():
        loader_tetrahedra = DataLoader(X_test['tetrahedra'], batch_size=batch_size)
        loader_triangles = DataLoader(X_test['triangles'], batch_size=batch_size)
        loader_simplex = DataLoader(X_test['simplex'], batch_size=batch_size)
        loader_targets = DataLoader(y_test, batch_size=batch_size)

        loaders = {"tetrahedra": loader_tetrahedra, "triangles": loader_triangles, 'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_test = CombinedLoader(loaders)

    if 'tetrahedra' in X_test.keys() and 'triangles' in X_test.keys() and 'simplex' not in X_test.keys():
        loader_tetrahedra = DataLoader(X_test['tetrahedra'], batch_size=batch_size)
        loader_triangles = DataLoader(X_test['triangles'], batch_size=batch_size)
        loader_targets = DataLoader(y_test, batch_size=batch_size)

        loaders = {"tetrahedra": loader_tetrahedra, "triangles": loader_triangles, 'targets': loader_targets}
        combined_loaders_test = CombinedLoader(loaders)

    if 'tetrahedra' in X_test.keys() and 'triangles' not in X_test.keys() and 'simplex' in X_test.keys():
        loader_tetrahedra = DataLoader(X_test['tetrahedra'], batch_size=batch_size)
        loader_simplex = DataLoader(X_test['simplex'], batch_size=batch_size)
        loader_targets = DataLoader(y_test, batch_size=batch_size)

        loaders = {"tetrahedra": loader_tetrahedra, 'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_test = CombinedLoader(loaders)

    if 'tetrahedra' in X_test.keys() and 'triangles' not in X_test.keys() and 'simplex' not in X_test.keys():
        loader_tetrahedra = DataLoader(X_test['tetrahedra'], batch_size=batch_size)
        loader_targets = DataLoader(y_test, batch_size=batch_size)

        loaders = {"tetrahedra": loader_tetrahedra, 'targets': loader_targets}
        combined_loaders_test = CombinedLoader(loaders)

    if 'tetrahedra' not in X_test.keys() and 'triangles' in X_test.keys() and 'simplex' in X_test.keys():
        loader_triangles = DataLoader(X_test['triangles'], batch_size=batch_size)
        loader_simplex = DataLoader(X_test['simplex'], batch_size=batch_size)
        loader_targets = DataLoader(y_test, batch_size=batch_size)

        loaders = {"triangles": loader_triangles, 'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_test = CombinedLoader(loaders)

    if 'tetrahedra' not in X_test.keys() and 'triangles' in X_test.keys() and 'simplex' not in X_test.keys():
        loader_triangles = DataLoader(X_test['triangles'], batch_size=batch_size)
        loader_targets = DataLoader(y_test, batch_size=batch_size)

        loaders = {"triangles": loader_triangles, 'targets': loader_targets}
        combined_loaders_test = CombinedLoader(loaders)

    if 'tetrahedra' not in X_test.keys() and 'triangles' not in X_test.keys() and 'simplex' in X_test.keys():
        loader_simplex = DataLoader(X_test['simplex'], batch_size=batch_size)
        loader_targets = DataLoader(y_test, batch_size=batch_size)

        loaders = {'simplex': loader_simplex, 'targets': loader_targets}
        combined_loaders_test = CombinedLoader(loaders)

    return combined_loaders_train, combined_loaders_val, combined_loaders_test


def batch_decomposition(batch):
    try:
        batch_tetrahedra = batch["tetrahedra"]
    except:
        batch_tetrahedra = None

    try:
        batch_triangles = batch["triangles"]
    except:
        batch_triangles = None

    try:
        batch_simplex = batch["simplex"]
    except:
        batch_simplex = None

    try:
        batch_targets = batch["targets"]
    except:
        batch_targets = None

    return batch_tetrahedra, batch_triangles, batch_simplex, batch_targets


def transform_outputs(outputs):
    preds_list = []
    targets_list = []
    for _ in outputs:
        preds = _['preds'].numpy().tolist()
        targets = _['targets'].numpy().tolist()
        preds_list.extend(preds)
        targets_list.extend(targets)
    return targets_list, preds_list
