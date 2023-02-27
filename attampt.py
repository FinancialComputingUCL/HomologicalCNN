from homological_models import *
from tmfg_bootstrapped import *
from tmfg_core import *
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import make_classification
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from skorch import NeuralNetClassifier


def data_classification(X, Y, T=1):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY


class Dataset(data.Dataset):

    def __init__(self, x_1, x_2, x_3, y, T=1):
        self.T = T

        try:
            x_1, y = data_classification(x_1, y)
        except:
            x_1 = None

        try:
            x_2, y = data_classification(x_2, y)
        except:
            x_2 = None

        try:
            x_3, y = data_classification(x_3, y)
        except:
            x_3 = None

        try:
            self.length = len(x_1)
        except:
            try:
                self.length = len(x_2)
            except:
                self.length = len(x_3)

        try:
            x_1 = torch.from_numpy(x_1)
        except:
            pass

        try:
            x_2 = torch.from_numpy(x_2)
        except:
            pass

        try:
            x_3 = torch.from_numpy(x_3)
        except:
            pass

        try:
            self.x_1 = torch.unsqueeze(x_1, 1)
        except:
            self.x_1 = None

        try:
            self.x_2 = torch.unsqueeze(x_2, 1)
        except:
            self.x_2 = None

        try:
            self.x_3 = torch.unsqueeze(x_3, 1)
        except:
            self.x_3 = None

        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        try:
            return_0 = self.x_1[index]
        except:
            return_0 = None

        try:
            return_1 = self.x_2[index]
        except:
            return_1 = None

        try:
            return_2 = self.x_3[index]
        except:
            return_2 = None

        if self.x_1 is not None and self.x_2 is not None and self.x_3 is not None:
            return return_0, return_1, return_2, self.y[index]
        elif self.x_1 is not None and self.x_2 is not None and self.x_3 is None:
            return return_0, return_1, self.y[index]
        elif self.x_1 is not None and self.x_2 is None and self.x_3 is not None:
            return return_0, return_2, self.y[index]
        elif self.x_1 is None and self.x_2 is not None and self.x_3 is not None:
            return return_1, return_2, self.y[index]


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=50, n_redundant=2)
    X = pd.DataFrame(X)
    X_test = X.iloc[-200:, :]
    y_test = y[-200:]
    X_val = X.iloc[-400:-200, :]
    y_val = y[-400:-200]
    X_train = X.iloc[:-400, :]
    y_train = y[:-400]

    x = None
    x_train = None
    x_val = None
    x_test = None

    cliques, separators, original_tmfg, _, adjacency_matrix = TMFG_Bootstrapped(X, 'pearson', 100, 90,
                                                                                parallel=True).compute_tmfg_bootstrapping()

    c = nx.degree_centrality(adjacency_matrix)

    keys = np.array(list(c.keys()))
    values = np.array(list(c.values()))
    nodes_list = sorted(list(keys[values != 0]))

    simplexes = []

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
                if (set(t).issubset(set(f))):
                    flag = True
            if flag == False:
                new_b_cliques_3.append(t)

    final_b_cliques_3 = new_b_cliques_3

    new_b_cliques_2 = []

    if len(final_b_cliques_3) == 0:
        new_b_cliques_2 = final_b_cliques_2

    else:
        for t in final_b_cliques_2:
            flag = False
            for f in final_b_cliques_3:
                if (set(t).issubset(set(f))):
                    flag = True
            if flag == False:
                new_b_cliques_2.append(t)

    final_b_cliques_2 = new_b_cliques_2

    new_b_cliques_2 = []

    if len(final_b_cliques_4) == 0:
        new_b_cliques_2 = final_b_cliques_2

    else:
        for t in final_b_cliques_2:
            flag = False
            for f in final_b_cliques_4:
                if (set(t).issubset(set(f))):
                    flag = True
            if flag == False:
                new_b_cliques_2.append(t)

    final_b_cliques_2 = new_b_cliques_2

    print(final_b_cliques_4)
    print(final_b_cliques_3)
    print(final_b_cliques_2)


    def get_final_X_4(X, final_b_cliques_4):

        final_X = None

        for e, c in enumerate(final_b_cliques_4):
            if final_X is None:
                final_X = pd.DataFrame()
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
                final_X = pd.DataFrame()
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
                final_X = pd.DataFrame()
                final_X = X[final_b_cliques_2[e]]
            else:
                final_X = pd.concat(
                    [final_X, X[final_b_cliques_2[e]]], ignore_index=True, axis=1
                )
        return final_X


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

    final_train_X_4_copy = copy.copy(final_train_X_4)
    final_train_X_3_copy = copy.copy(final_train_X_3)
    final_train_X_2_copy = copy.copy(final_train_X_2)

    final_val_X_4_copy = copy.copy(final_val_X_4)
    final_val_X_3_copy = copy.copy(final_val_X_3)
    final_val_X_2_copy = copy.copy(final_val_X_2)

    final_test_X_4_copy = copy.copy(final_test_X_4)
    final_test_X_3_copy = copy.copy(final_test_X_3)
    final_test_X_2_copy = copy.copy(final_test_X_2)

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

    model = HCNN_model(T=1, FILTERS_L1=4, FILTERS_L2=32,
                       last_layer_neurons=len(pd.Series(y_train).unique()),
                       NF_4=shape_4, NF_3=shape_3, NF_2=shape_2)

    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    batch_size = 64

    dataset_train = Dataset(final_train_X_4, final_train_X_3, final_train_X_2, y_train)
    dataset_val = Dataset(final_val_X_4, final_val_X_3, final_val_X_2, y_val)
    dataset_test = Dataset(final_test_X_4, final_test_X_3, final_test_X_2, y_test)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):

        train_losses = np.zeros(epochs)
        test_losses = np.zeros(epochs)
        best_test_loss = np.inf
        best_test_epoch = 0

        for it in tqdm(range(epochs)):

            model.train()
            train_loss = []
            for i in train_loader:
                print(shape_4)
                print(shape_3)
                print(shape_2)
                # move data to GPU
                if shape_4 is not None and shape_3 is not None and shape_2 is not None:
                    tetrahedra = i[0].to(device='cpu', dtype=torch.float)
                    triangles = i[1].to(device='cpu', dtype=torch.float)
                    simplex = i[2].to(device='cpu', dtype=torch.float)
                    targets = i[3].to(device='cpu', dtype=torch.float)
                elif shape_4 is not None and shape_3 is not None and shape_2 is None:
                    tetrahedra = i[0].to(device='cpu', dtype=torch.float)
                    triangles = i[1].to(device='cpu', dtype=torch.float)
                    simplex = None
                    targets = i[2].to(device='cpu', dtype=torch.float)
                elif shape_4 is not None and shape_3 is None and shape_2 is not None:
                    tetrahedra = i[0].to(device='cpu', dtype=torch.float)
                    triangles = None
                    simplex = i[1].to(device='cpu', dtype=torch.float)
                    targets = i[2].to(device='cpu', dtype=torch.float)
                elif shape_4 is None and shape_3 is not None and shape_2 is not None:
                    tetrahedra = None
                    triangles = i[0].to(device='cpu', dtype=torch.float)
                    simplex = i[1].to(device='cpu', dtype=torch.float)
                    targets = i[2].to(device='cpu', dtype=torch.float)

                # print("inputs.shape:", inputs.shape)
                # zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                # print("about to get model output")
                outputs = model(tetrahedra, triangles, simplex)
                # print("done getting model output")
                # print("outputs.shape:", outputs.shape, "targets.shape:", targets.shape)
                loss = criterion(outputs, targets.long())
                # Backward and optimize
                # print("about to optimize")
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            # Get train loss and test loss
            train_loss = np.mean(train_loss)  # a little misleading

            # Save losses
            train_losses[it] = train_loss

            print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, Best Val Epoch: {best_test_epoch}')

        return train_losses

    train_losses = batch_gd(model, criterion, optimizer, train_loader, val_loader, epochs=50)

    ####
    '''net = NeuralNetClassifier(
        model,
        criterion=criterion,
        max_epochs=100,
    )

    try:
        final_train_X_4 = final_train_X_4.reshape(final_train_X_4.shape[0], final_train_X_4.shape[1])
    except:
        pass

    try:
        final_train_X_3 = final_train_X_3.reshape(final_train_X_3.shape[0], final_train_X_3.shape[1])
    except:
        pass

    try:
        final_train_X_2 = final_train_X_2.reshape(final_train_X_2.shape[0], final_train_X_2.shape[1])
    except:
        pass

    if final_train_X_4 is not None and final_train_X_3 is not None and final_train_X_2 is not None:
        X = {'tetrahedra': final_train_X_4, 'triangles': final_train_X_3, 'simplex': final_train_X_2}
    if final_train_X_4 is None and final_train_X_3 is not None and final_train_X_2 is not None:
        X = {'triangles': final_train_X_3, 'simplex': final_train_X_2}
    if final_train_X_4 is not None and final_train_X_3 is None and final_train_X_2 is not None:
        X = {'tetrahedra': final_train_X_4, 'simplex': final_train_X_2}
    if final_train_X_4 is not None and final_train_X_3 is None and final_train_X_2 is None:
        X = {'tetrahedra': final_train_X_4}
    if final_train_X_4 is None and final_train_X_3 is not None and final_train_X_2 is None:
        X = {'triangles': final_train_X_3}
    if final_train_X_4 is None and final_train_X_3 is None and final_train_X_2 is not None:
        X = {'simplex': final_train_X_2}

    print(X)
    print(y_train)
    print(net)
    net.fit(X, y_train)
'''