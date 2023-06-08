from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

from homological_models import *


class HCNN:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test,
                 n_filters_l1, n_filters_l2, tmfg_repetitions, tmfg_confidence,
                 tmfg_similarity, dropout_rate, max_epochs=params.MAX_EPOCHS, T=1, root_folder=None, filtering_type=None, seed=None):
        self.number_of_selected_features, self.shape_tetrahedra, self.shape_triangles, self.shape_simplex, self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = h_input_transform(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            tmfg_repetitions=tmfg_repetitions,
            tmfg_confidence=tmfg_confidence,
            tmfg_similarity=tmfg_similarity,
            filtering_type=filtering_type)

        self.T = T
        self.n_filters_l1 = n_filters_l1
        self.n_filters_l2 = n_filters_l2
        self.last_layer_neurons = len(pd.Series(y_train).unique())

        self.best_model = None

        self.model = HCNN_model1D(T=self.T,
                                  FILTERS_L1=self.n_filters_l1,
                                  FILTERS_L2=self.n_filters_l2,
                                  last_layer_neurons=self.last_layer_neurons,
                                  NF_4=self.shape_tetrahedra,
                                  NF_3=self.shape_triangles,
                                  NF_2=self.shape_simplex,
                                  dropout_rate=dropout_rate,)

        # Set the global seed for reproducibility.
        pl.seed_everything(seed)

        # Set learning rate based on OpenAI's implementation.
        self.model.set_lr(get_openai_lr(self.model))

        self.early_stopping = EarlyStopping('val_loss', mode='min', patience=params.EARLY_STOPPING_PATIENCE)
        self.checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

        self.logger_folder = root_folder + 'logs'

        if params.DEVICE == 'cuda':
            self.trainer = Trainer(max_epochs=max_epochs,
                                   accelerator='gpu',
                                   enable_progress_bar=False,
                                   enable_model_summary=False,
                                   callbacks=[self.early_stopping, self.checkpoint_callback],
                                   default_root_dir=self.logger_folder,
                                   )
        else:
            self.trainer = Trainer(max_epochs=max_epochs,
                                   enable_progress_bar=False,
                                   enable_model_summary=False,
                                   callbacks=[self.early_stopping, self.checkpoint_callback],
                                   default_root_dir=self.logger_folder,
                                   )

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
        self.trainer.validate(self.model, dataloaders=self.val_dataloader, verbose=False, ckpt_path='best')
        return self.model.val_targets, self.model.val_preds, self.model.val_probs

    def predict(self):
        self.trainer.test(self.model, dataloaders=self.test_dataloader, verbose=False, ckpt_path='best')
        return self.model.test_targets, self.model.test_preds, self.model.test_probs

    def data_preparation_pipeline(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.model, self.number_of_selected_features
