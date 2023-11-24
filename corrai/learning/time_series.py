from abc import ABC, abstractmethod

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def plot_sequence_forcast(X, y_ref, model, X_target_index=0, batch_nb=0):
    """
    Plot the historical, reference, and predicted values for a given
    batch number in X.

    Parameters:
    :param X: np.ndarray 3D array of shape [batch_size, time_step, dimension]
    Input sequences.
    :param y_ref: np.ndarray of shape [batch_size, time_step].
    :param model: keras.Model The trained model for making predictions.
    :param X_target_index: The index of 3rd dimension in X that contains
        historical values.
    :param batch_nb: int Batch number to visualize. Default is 0.
    """
    predictions = model.predict(X)
    predictions = predictions[batch_nb, :]
    y_ref_to_plot = y_ref[batch_nb, :]
    x_to_plot = X[batch_nb, :, X_target_index]

    fig, ax = plt.subplots()
    # Plot historic values
    ax.plot(
        range(len(x_to_plot)),
        x_to_plot,
        marker="o",
        label="Historic Values",
    )

    # Plot new values
    ax.plot(
        range(len(x_to_plot), len(x_to_plot) + len(y_ref_to_plot)),
        y_ref_to_plot,
        marker="o",
        label="New Values",
    )

    # Plot predicted values
    ax.plot(
        range(len(x_to_plot), len(x_to_plot) + len(y_ref_to_plot)),
        predictions,
        marker="o",
        label="Predicted Values",
    )

    # Set labels and title
    ax.set_xlabel("Number of Timesteps")
    ax.set_ylabel("x(t)")
    ax.set_title("Actual and Predicted Values Over Time")
    ax.legend()

    plt.show()


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
