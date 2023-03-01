from pytorch_lightning import Trainer

import params
from homological_models import *


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

        self.model = HCNN_model1D(T=T,
                                  FILTERS_L1=n_filters_l1,
                                  FILTERS_L2=n_filters_l2,
                                  last_layer_neurons=len(pd.Series(y_train).unique()),
                                  NF_4=self.shape_tetrahedra,
                                  NF_3=self.shape_triangles,
                                  NF_2=self.shape_simplex)

        learning_rate = get_openai_lr(self.model)
        self.model.set_lr(learning_rate)

        self.trainer = Trainer(max_epochs=3)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = prepare_dataloaders(self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test)

    def fit(self):
        self.trainer.fit(self.model, self.train_dataloader)

    def evaluate(self):
        self.model.eval()
        self.trainer = Trainer()
        self.trainer.validate(self.model, dataloaders=self.val_dataloader)
        return self.model.val_targets, self.model.val_preds

    def predict(self):
        self.model.eval()
        self.trainer = Trainer()
        self.trainer.test(self.model, dataloaders=self.test_dataloader)
        return self.model.test_targets, self.model.test_preds

    def data_preparation_pipeline(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.model, self.number_of_selected_features
