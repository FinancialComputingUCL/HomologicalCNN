from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

import params
from homological_models import *


class HCNN:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test,
                 n_filters_l1, n_filters_l2, tmfg_repetitions, tmfg_confidence,
                 tmfg_similarity, max_epochs=params.MAX_EPOCHS, T=1):
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

        # Set learning rate based on OpenAI's implementation.
        self.model.set_lr(get_openai_lr(self.model))

        early_stopping = EarlyStopping('val_loss')
        if params.DEVICE == 'cuda':
            self.trainer = Trainer(max_epochs=max_epochs, accelerator='gpu', enable_progress_bar=False, enable_model_summary=False, callbacks=[early_stopping])
        else:
            self.trainer = Trainer(max_epochs=max_epochs, enable_progress_bar=False, enable_model_summary=False, callbacks=[early_stopping])

        self.train_dataloader, self.val_dataloader, self.test_dataloader = prepare_dataloaders(self.X_train,
                                                                                               self.X_val,
                                                                                               self.X_test,
                                                                                               self.y_train,
                                                                                               self.y_val,
                                                                                               self.y_test,
                                                                                               batch_size=params.BATCH_SIZE)

    def fit(self):
        self.trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

    def evaluate(self):
        self.model.eval()
        self.trainer = Trainer()
        self.trainer.validate(self.model, dataloaders=self.val_dataloader, verbose=False)
        return self.model.val_targets, self.model.val_preds, self.model.val_probs

    def predict(self):
        self.model.eval()
        self.trainer = Trainer()
        self.trainer.test(self.model, dataloaders=self.test_dataloader, verbose=False)
        return self.model.test_targets, self.model.test_preds, self.model.test_probs

    def data_preparation_pipeline(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.model, self.number_of_selected_features
