from homological_models import *
import params

from skorch.callbacks import EarlyStopping


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

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        device = torch.device(params.DEVICE if torch.cuda.is_available() else "cpu")
        print("Using device: ", device)

        self.net = NeuralNetClassifier(
            self.model,
            criterion=self.criterion,
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(patience=params.EARLY_STOPPING_PATIENCE)],
            batch_size=params.BATCH_SIZE,
            verbose=0,
            device=device,
        )

    def data_preparation_pipeline(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.net, self.number_of_selected_features
