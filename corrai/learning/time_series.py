from abc import ABC, abstractmethod

import keras
from sklearn.base import BaseEstimator, RegressorMixin


class KerasModelSkBC(ABC, BaseEstimator, RegressorMixin):
    def __init__(
        self,
        loss=None,
        optimizer=None,
        max_epoch: int = 20,
        patience: int = 2,
        metrics=None,
    ):
        self.max_epoch = max_epoch
        self.patience = patience
        self.metrics = metrics
        self.loss = loss if loss is not None else keras.losses.MeanSquaredError()
        self.optimizer = optimizer if optimizer is not None else keras.optimizers.Adam()

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def _main_fit(self, model, X, y, x_val=None, y_val=None):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, mode="min"
        )

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metrics])

        validation_data = (
            (x_val, y_val) if x_val is not None and y_val is not None else None
        )

        history = model.fit(
            X,
            y,
            epochs=self.max_epoch,
            validation_data=validation_data,
            callbacks=[early_stopping],
        )

        self.history = history
        self._is_fitted = True
        self.model = model

    @abstractmethod
    def fit(self, X, y, x_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class TsLinearModel(KerasModelSkBC):
    def __init__(
        self,
        loss=None,
        optimizer=None,
        max_epoch: int = 20,
        patience: int = 2,
        metrics=None,
    ):
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            max_epoch=max_epoch,
            patience=patience,
            metrics=metrics,
        )

    def fit(self, X, y, x_val=None, y_val=None):
        model = keras.models.Sequential(
            [
                keras.layers.Flatten(input_shape=[X.shape[1] * X.shape[2], 1]),
                keras.layers.Dense(y.shape[1]),
            ]
        )

        X = X.reshape(X.shape[0], -1)
        if x_val is not None:
            x_val = x_val.reshape(x_val.shape[0], -1)

        self._main_fit(model, X, y, x_val, y_val)

    def predict(self, X):
        X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)
