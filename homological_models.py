import torch
import torch.nn as nn

from skorch import NeuralNetClassifier
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from homological_utils import *
import params


class HCNN_model2D(nn.Module):
    def __init__(self, T, FILTERS_L1, FILTERS_L2, last_layer_neurons, NF_4=None, NF_3=None, NF_2=None):
        super().__init__()
        self.NF_4 = NF_4
        self.NF_3 = NF_3
        self.NF_2 = NF_2

        if (self.NF_4 is not None) and (self.NF_3 is not None) and (self.NF_2 is not None):
            self.logic_conv_tetrahedra = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 4),
                          stride=(1, 4)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_4 / 4))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_conv_triangles = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 3),
                          stride=(1, 3)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_3 / 3))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_conv_simplex = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 2),
                          stride=(1, 2)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_2 / 2))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2) + 32)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 32), out_features=((FILTERS_L2 * 2) + 64)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is None) and (self.NF_3 is not None) and (self.NF_2 is not None):

            self.logic_conv_triangles = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 3),
                          stride=(1, 3)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_3 / 3))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_conv_simplex = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 2),
                          stride=(1, 2)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_2 / 2))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2) + 32)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 32), out_features=((FILTERS_L2 * 2) + 64)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is None) and (self.NF_3 is None) and (self.NF_2 is not None):

            self.logic_conv_simplex = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 2),
                          stride=(1, 2)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_2 / 2))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2) + 32)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 32), out_features=((FILTERS_L2 * 2) + 64)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is not None) and (self.NF_3 is None) and (self.NF_2 is None):
            self.logic_conv_tetrahedra = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 4),
                          stride=(1, 4)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_4 / 4))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2) + 32)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 32), out_features=((FILTERS_L2 * 2) + 64)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is not None) and (self.NF_3 is None) and (self.NF_2 is not None):
            self.logic_conv_tetrahedra = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 4),
                          stride=(1, 4)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_4 / 4))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_conv_simplex = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 2),
                          stride=(1, 2)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_2 / 2))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2) + 32)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 32), out_features=((FILTERS_L2 * 2) + 64)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is None) and (self.NF_3 is not None) and (self.NF_2 is None):
            self.logic_conv_triangles = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 3),
                          stride=(1, 3)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_3 / 3))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2) + 32)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 32), out_features=((FILTERS_L2 * 2) + 64)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is not None) and (self.NF_3 is not None) and (self.NF_2 is None):
            self.logic_conv_tetrahedra = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 4),
                          stride=(1, 4)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_4 / 4))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_conv_triangles = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=(1, 3),
                          stride=(1, 3)),
                nn.BatchNorm2d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(1, int(NF_3 / 3))),
                nn.BatchNorm2d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2) + 32)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 32), out_features=((FILTERS_L2 * 2) + 64)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 64), out_features=last_layer_neurons)
            )

    def forward(self, tetrahedra=None, triangles=None, simplex=None):

        if (self.NF_4 is not None) and (self.NF_3 is not None) and (self.NF_2 is not None):
            tetrahedra = tetrahedra.reshape(tetrahedra.shape[0], 1, 1, tetrahedra.shape[1]).float()
            triangles = triangles.reshape(triangles.shape[0], 1, 1, triangles.shape[1]).float()
            simplex = simplex.reshape(simplex.shape[0], 1, 1, simplex.shape[1]).float()

            tetrahedra = self.logic_conv_tetrahedra(tetrahedra)
            triangles = self.logic_conv_triangles(triangles)
            simplex = self.logic_conv_simplex(simplex)

            tetrahedra = torch.reshape(tetrahedra, (tetrahedra.shape[0], tetrahedra.shape[1]))
            triangles = torch.reshape(triangles, (triangles.shape[0], triangles.shape[1]))
            simplex = torch.reshape(simplex, (simplex.shape[0], simplex.shape[1]))

            concatenation = torch.cat([tetrahedra, triangles, simplex], dim=1)

            x = self.logic_mlp(concatenation)
            preds = torch.softmax(x, dim=1)

            return preds

        elif (self.NF_4 is None) and (self.NF_3 is not None) and (self.NF_2 is not None):

            triangles = triangles.reshape(triangles.shape[0], 1, 1, triangles.shape[1]).float()
            simplex = simplex.reshape(simplex.shape[0], 1, 1, simplex.shape[1]).float()

            triangles = self.logic_conv_triangles(triangles)
            simplex = self.logic_conv_simplex(simplex)

            triangles = torch.reshape(triangles, (triangles.shape[0], triangles.shape[1]))
            simplex = torch.reshape(simplex, (simplex.shape[0], simplex.shape[1]))

            concatenation = torch.cat([triangles, simplex], dim=1)

            x = self.logic_mlp(concatenation)
            preds = torch.softmax(x, dim=1)

            return preds

        elif (self.NF_4 is None) and (self.NF_3 is None) and (self.NF_2 is not None):
            simplex = simplex.reshape(simplex.shape[0], 1, 1, simplex.shape[1]).float()

            simplex = self.logic_conv_simplex(simplex)

            simplex = torch.reshape(simplex, (simplex.shape[0], simplex.shape[1]))

            x = self.logic_mlp(simplex)
            preds = torch.softmax(x, dim=1)

            return preds

        elif (self.NF_4 is not None) and (self.NF_3 is None) and (self.NF_2 is None):
            tetrahedra = tetrahedra.reshape(tetrahedra.shape[0], 1, 1, tetrahedra.shape[1]).float()

            tetrahedra = self.logic_conv_tetrahedra(tetrahedra)

            tetrahedra = torch.reshape(tetrahedra, (tetrahedra.shape[0], tetrahedra.shape[1]))

            x = self.logic_mlp(tetrahedra)
            preds = torch.softmax(x, dim=1)

            return preds

        elif (self.NF_4 is not None) and (self.NF_3 is None) and (self.NF_2 is not None):
            tetrahedra = tetrahedra.reshape(tetrahedra.shape[0], 1, 1, tetrahedra.shape[1]).float()
            simplex = simplex.reshape(simplex.shape[0], 1, 1, simplex.shape[1]).float()

            tetrahedra = self.logic_conv_tetrahedra(tetrahedra)
            simplex = self.logic_conv_simplex(simplex)

            tetrahedra = torch.reshape(tetrahedra, (tetrahedra.shape[0], tetrahedra.shape[1]))
            simplex = torch.reshape(simplex, (simplex.shape[0], simplex.shape[1]))

            concatenation = torch.cat([tetrahedra, simplex], dim=1)

            x = self.logic_mlp(concatenation)
            preds = torch.softmax(x, dim=1)

            return preds

        elif (self.NF_4 is None) and (self.NF_3 is not None) and (self.NF_2 is None):
            triangles = triangles.reshape(triangles.shape[0], 1, 1, triangles.shape[1]).float()

            triangles = self.logic_conv_triangles(triangles)

            triangles = torch.reshape(triangles, (triangles.shape[0], triangles.shape[1]))

            x = self.logic_mlp(triangles)
            preds = torch.softmax(x, dim=1)

            return preds

        elif (self.NF_4 is not None) and (self.NF_3 is not None) and (self.NF_2 is None):
            tetrahedra = tetrahedra.reshape(tetrahedra.shape[0], 1, 1, tetrahedra.shape[1]).float()
            triangles = triangles.reshape(triangles.shape[0], 1, 1, triangles.shape[1]).float()

            tetrahedra = self.logic_conv_tetrahedra(tetrahedra)
            triangles = self.logic_conv_triangles(triangles)

            tetrahedra = torch.reshape(tetrahedra, (tetrahedra.shape[0], tetrahedra.shape[1]))
            triangles = torch.reshape(triangles, (triangles.shape[0], triangles.shape[1]))

            concatenation = torch.cat([tetrahedra, triangles], dim=1)

            x = self.logic_mlp(concatenation)
            preds = torch.softmax(x, dim=1)

            return preds


class HCNN_model1D(LightningModule):
    def __init__(self, T, FILTERS_L1, FILTERS_L2, last_layer_neurons, NF_4=None, NF_3=None, NF_2=None, lr=0.001):
        super().__init__()
        self.NF_4 = NF_4
        self.NF_3 = NF_3
        self.NF_2 = NF_2

        self.val_targets = None
        self.val_preds = None
        self.val_probs = None
        self.test_targets = None
        self.test_preds = None
        self.test_probs = None
        self.lr = lr

        if (self.NF_4 is not None) and (self.NF_3 is not None) and (self.NF_2 is not None):
            self.logic_conv_tetrahedra = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=4,
                          stride=4),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_4 / 4))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_conv_triangles = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=3,
                          stride=3),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_3 / 3))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_conv_simplex = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=2,
                          stride=2),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_2 / 2))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=(FILTERS_L2 * 3), out_features=((FILTERS_L2 * 3)+32)),
                nn.Linear(in_features=((FILTERS_L2 * 3)+32), out_features=((FILTERS_L2 * 3)+64)),
                nn.Linear(in_features=((FILTERS_L2 * 3)+64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is None) and (self.NF_3 is not None) and (self.NF_2 is not None):

            self.logic_conv_triangles = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=3,
                          stride=3),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_3 / 3))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_conv_simplex = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=2,
                          stride=2),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_2 / 2))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2)+32)),
                nn.Linear(in_features=((FILTERS_L2 * 2)+32), out_features=((FILTERS_L2 * 2)+64)),
                nn.Linear(in_features=((FILTERS_L2 * 2)+64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is None) and (self.NF_3 is None) and (self.NF_2 is not None):

            self.logic_conv_simplex = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=2,
                          stride=2),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_2 / 2))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=FILTERS_L2, out_features=(FILTERS_L2+32)),
                nn.Linear(in_features=(FILTERS_L2+32), out_features=(FILTERS_L2+64)),
                nn.Linear(in_features=(FILTERS_L2+64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is not None) and (self.NF_3 is None) and (self.NF_2 is None):
            self.logic_conv_tetrahedra = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=4,
                          stride=4),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_4 / 4))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=FILTERS_L2, out_features=(FILTERS_L2 + 32)),
                nn.Linear(in_features=(FILTERS_L2 + 32), out_features=(FILTERS_L2 + 64)),
                nn.Linear(in_features=(FILTERS_L2 + 64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is not None) and (self.NF_3 is None) and (self.NF_2 is not None):
            self.logic_conv_tetrahedra = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=4,
                          stride=4),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_4 / 4))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_conv_simplex = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=2,
                          stride=2),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_2 / 2))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2) + 32)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 32), out_features=((FILTERS_L2 * 2) + 64)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is None) and (self.NF_3 is not None) and (self.NF_2 is None):
            self.logic_conv_triangles = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=3,
                          stride=3),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_3 / 3))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=FILTERS_L2, out_features=(FILTERS_L2 + 32)),
                nn.Linear(in_features=(FILTERS_L2 + 32), out_features=(FILTERS_L2 + 64)),
                nn.Linear(in_features=(FILTERS_L2 + 64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is not None) and (self.NF_3 is not None) and (self.NF_2 is None):
            self.logic_conv_tetrahedra = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=4,
                          stride=4),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_4 / 4))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_conv_triangles = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=3,
                          stride=3),
                nn.BatchNorm1d(FILTERS_L1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(in_channels=FILTERS_L1,
                          out_channels=FILTERS_L2,
                          kernel_size=(int(NF_3 / 3))),
                nn.BatchNorm1d(FILTERS_L2),
                nn.LeakyReLU(negative_slope=0.01),
            )

            self.logic_mlp = nn.Sequential(
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2) + 32)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 32), out_features=((FILTERS_L2 * 2) + 64)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 64), out_features=last_layer_neurons)
            )

    def forward(self, tetrahedra=None, triangles=None, simplex=None):

        if (self.NF_4 is not None) and (self.NF_3 is not None) and (self.NF_2 is not None):
            tetrahedra = tetrahedra.view(tetrahedra.shape[0], 1, tetrahedra.shape[1]).float()
            triangles = triangles.view(triangles.shape[0], 1, triangles.shape[1]).float()
            simplex = simplex.view(simplex.shape[0], 1, simplex.shape[1]).float()

            tetrahedra = self.logic_conv_tetrahedra(tetrahedra)
            triangles = self.logic_conv_triangles(triangles)
            simplex = self.logic_conv_simplex(simplex)

            tetrahedra = torch.reshape(tetrahedra, (tetrahedra.shape[0], tetrahedra.shape[1]))
            triangles = torch.reshape(triangles, (triangles.shape[0], triangles.shape[1]))
            simplex = torch.reshape(simplex, (simplex.shape[0], simplex.shape[1]))

            concatenation = torch.cat([tetrahedra, triangles, simplex], dim=1)

            x = self.logic_mlp(concatenation)
            preds = torch.log_softmax(x, dim=1)

            return preds

        elif (self.NF_4 is None) and (self.NF_3 is not None) and (self.NF_2 is not None):

            triangles = triangles.view(triangles.shape[0], 1, triangles.shape[1]).float()
            simplex = simplex.view(simplex.shape[0], 1, simplex.shape[1]).float()

            triangles = self.logic_conv_triangles(triangles)
            simplex = self.logic_conv_simplex(simplex)

            triangles = torch.reshape(triangles, (triangles.shape[0], triangles.shape[1]))
            simplex = torch.reshape(simplex, (simplex.shape[0], simplex.shape[1]))

            concatenation = torch.cat([triangles, simplex], dim=1)

            x = self.logic_mlp(concatenation)
            preds = torch.log_softmax(x, dim=1)

            return preds

        elif (self.NF_4 is None) and (self.NF_3 is None) and (self.NF_2 is not None):
            simplex = simplex.view(simplex.shape[0], 1, simplex.shape[1]).float()

            simplex = self.logic_conv_simplex(simplex)

            simplex = torch.reshape(simplex, (simplex.shape[0], simplex.shape[1]))

            x = self.logic_mlp(simplex)
            preds = torch.log_softmax(x, dim=1)

            return preds

        elif (self.NF_4 is not None) and (self.NF_3 is None) and (self.NF_2 is None):
            tetrahedra = tetrahedra.view(tetrahedra.shape[0], 1, tetrahedra.shape[1]).float()

            tetrahedra = self.logic_conv_tetrahedra(tetrahedra)

            tetrahedra = torch.reshape(tetrahedra, (tetrahedra.shape[0], tetrahedra.shape[1]))

            x = self.logic_mlp(tetrahedra)
            preds = torch.log_softmax(x, dim=1)

            return preds

        elif (self.NF_4 is not None) and (self.NF_3 is None) and (self.NF_2 is not None):
            tetrahedra = tetrahedra.view(tetrahedra.shape[0], 1, tetrahedra.shape[1]).float()
            simplex = simplex.view(simplex.shape[0], 1, simplex.shape[1]).float()

            tetrahedra = self.logic_conv_tetrahedra(tetrahedra)
            simplex = self.logic_conv_simplex(simplex)

            tetrahedra = torch.reshape(tetrahedra, (tetrahedra.shape[0], tetrahedra.shape[1]))
            simplex = torch.reshape(simplex, (simplex.shape[0], simplex.shape[1]))

            concatenation = torch.cat([tetrahedra, simplex], dim=1)

            x = self.logic_mlp(concatenation)
            preds = torch.log_softmax(x, dim=1)

            return preds

        elif (self.NF_4 is None) and (self.NF_3 is not None) and (self.NF_2 is None):
            triangles = triangles.view(triangles.shape[0], 1, triangles.shape[1]).float()

            triangles = self.logic_conv_triangles(triangles)

            triangles = torch.reshape(triangles, (triangles.shape[0], triangles.shape[1]))

            x = self.logic_mlp(triangles)
            preds = torch.log_softmax(x, dim=1)

            return preds

        elif (self.NF_4 is not None) and (self.NF_3 is not None) and (self.NF_2 is None):
            tetrahedra = tetrahedra.view(tetrahedra.shape[0], 1, tetrahedra.shape[1]).float()
            triangles = triangles.view(triangles.shape[0], 1, triangles.shape[1]).float()

            tetrahedra = self.logic_conv_tetrahedra(tetrahedra)
            triangles = self.logic_conv_triangles(triangles)

            tetrahedra = torch.reshape(tetrahedra, (tetrahedra.shape[0], tetrahedra.shape[1]))
            triangles = torch.reshape(triangles, (triangles.shape[0], triangles.shape[1]))

            concatenation = torch.cat([tetrahedra, triangles], dim=1)

            x = self.logic_mlp(concatenation)
            preds = torch.log_softmax(x, dim=1)

            return preds

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        batch_tetrahedra, batch_triangles, batch_simplex, batch_targets = batch_decomposition(train_batch)
        logits = self.forward(batch_tetrahedra, batch_triangles, batch_simplex)
        loss = self.cross_entropy_loss(logits, batch_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        batch_tetrahedra, batch_triangles, batch_simplex, batch_targets = batch_decomposition(val_batch)
        logits = self.forward(batch_tetrahedra, batch_triangles, batch_simplex)
        preds = torch.argmax(logits, dim=1)
        loss = self.cross_entropy_loss(logits, batch_targets)
        self.log('val_loss', loss)
        return {'probs': torch.exp(logits), 'preds': preds, 'targets': batch_targets}

    def test_step(self, test_batch, batch_idx):
        batch_tetrahedra, batch_triangles, batch_simplex, batch_targets = batch_decomposition(test_batch)
        logits = self.forward(batch_tetrahedra, batch_triangles, batch_simplex)
        preds = torch.argmax(logits, dim=1)
        loss = self.cross_entropy_loss(logits, batch_targets)
        self.log('test_loss', loss)
        return {'probs': torch.exp(logits), 'preds': preds, 'targets': batch_targets}

    def test_epoch_end(self, outputs):
        targets, preds, probs = transform_outputs(outputs)
        self.test_targets = targets
        self.test_preds = preds
        self.test_probs = probs

    def validation_epoch_end(self, outputs):
        targets, preds, probs = transform_outputs(outputs)
        self.val_targets = targets
        self.val_preds = preds
        self.val_probs = probs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.MAX_EPOCHS//4)
        return [optimizer], [scheduler]

    def set_lr(self, lr):
        self.lr = lr

