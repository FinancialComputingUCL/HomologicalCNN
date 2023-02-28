from pytorch_lightning import Trainer

from homological_models import *
import params

from skorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader

from torch import Tensor


class HCNN:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test,
                 n_filters_l1, n_filters_l2, tmfg_repetitions, tmfg_confidence,
                 tmfg_similarity, learning_rate, max_epochs=params.MAX_EPOCHS, T=1):
        self.number_of_selected_features, self.shape_tetrahedra, self.shape_triangles, self.shape_simplex, self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = h_input_transform(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            tmfg_repetitions=tmfg_repetitions,
            tmfg_confidence=tmfg_confidence,
            tmfg_similarity=tmfg_similarity
        )

        print(self.X_train)
        print(self.X_val)
        print(self.X_test)

        self.model = HCNN_model1D(T=T,
                                  FILTERS_L1=n_filters_l1,
                                  FILTERS_L2=n_filters_l2,
                                  last_layer_neurons=len(pd.Series(y_train).unique()),
                                  NF_4=self.shape_tetrahedra,
                                  NF_3=self.shape_triangles,
                                  NF_2=self.shape_simplex)

        self.net = self.model
        self.trainer = Trainer(max_epochs=3)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = prepare_dataloaders(self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test)

    def fit(self):
        self.trainer.fit(self.net, self.train_dataloader)

    def predict(self):
        self.net.eval()
        self.trainer = Trainer()
        self.trainer.test(self.net, dataloaders=self.test_dataloader)
        return self.net.test_targets, self.net.test_preds

    def data_preparation_pipeline(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.net, self.number_of_selected_features
