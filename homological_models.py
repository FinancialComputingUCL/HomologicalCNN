import torch
import torch.nn as nn

from skorch import NeuralNetClassifier

from homological_utils import *


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


class HCNN_model1D(nn.Module):
    def __init__(self, T, FILTERS_L1, FILTERS_L2, last_layer_neurons, NF_4=None, NF_3=None, NF_2=None):
        super().__init__()
        self.NF_4 = NF_4
        self.NF_3 = NF_3
        self.NF_2 = NF_2

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
                nn.Linear(in_features=FILTERS_L2, out_features=128),
                nn.Linear(in_features=128, out_features=64),
                nn.Linear(in_features=64, out_features=last_layer_neurons)
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
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2) + 32)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 32), out_features=((FILTERS_L2 * 2) + 64)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 64), out_features=last_layer_neurons)
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
                nn.Linear(in_features=(FILTERS_L2 * 2), out_features=((FILTERS_L2 * 2) + 32)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 32), out_features=((FILTERS_L2 * 2) + 64)),
                nn.Linear(in_features=((FILTERS_L2 * 2) + 64), out_features=last_layer_neurons)
            )

        elif (self.NF_4 is not None) and (self.NF_3 is not None) and (self.NF_2 is None):
            self.logic_conv_tetrahedra = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=FILTERS_L1,
                          kernel_size=4,
                          stride=4),
                nn.BatchNorm2d(FILTERS_L1),
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
            preds = torch.softmax(x, dim=1)

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
            preds = torch.softmax(x, dim=1)

            return preds

        elif (self.NF_4 is None) and (self.NF_3 is None) and (self.NF_2 is not None):
            simplex = simplex.view(simplex.shape[0], 1, simplex.shape[1]).float()

            simplex = self.logic_conv_simplex(simplex)

            simplex = torch.reshape(simplex, (simplex.shape[0], simplex.shape[1]))

            x = self.logic_mlp(simplex)
            preds = torch.softmax(x, dim=1)

            return preds

        elif (self.NF_4 is not None) and (self.NF_3 is None) and (self.NF_2 is None):
            tetrahedra = tetrahedra.view(tetrahedra.shape[0], 1, tetrahedra.shape[1]).float()

            tetrahedra = self.logic_conv_tetrahedra(tetrahedra)

            tetrahedra = torch.reshape(tetrahedra, (tetrahedra.shape[0], tetrahedra.shape[1]))

            x = self.logic_mlp(tetrahedra)
            preds = torch.softmax(x, dim=1)

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
            preds = torch.softmax(x, dim=1)

            return preds

        elif (self.NF_4 is None) and (self.NF_3 is not None) and (self.NF_2 is None):
            triangles = triangles.view(triangles.shape[0], 1, triangles.shape[1]).float()

            triangles = self.logic_conv_triangles(triangles)

            triangles = torch.reshape(triangles, (triangles.shape[0], triangles.shape[1]))

            x = self.logic_mlp(triangles)
            preds = torch.softmax(x, dim=1)

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
            preds = torch.softmax(x, dim=1)

            return preds