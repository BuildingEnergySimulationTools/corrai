from abc import ABC, abstractmethod

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def reshape_target_sequence_to_sequence(X, y, X_idx_target=0):
    """
    Reshapes the target sequence and concatenates it with the input sequence along
    the time axis.
    This function shall be used to turn a sequence to vector ML model to a sequence
    to sequence model.
    Code is adapted from the book "Hands-On Machine Learning with Scikit-Learn,
    Keras & TensorFlow" A. Géron, O'Reilly

    Parameters:
    :param X (numpy.ndarray): Input sequence array with shape
        [batch_sqize, n_step_history, features].
    :param y (numpy.ndarray): Target sequence array with shape
        [batch_size, n_step_future].
    :param X_idx_target (int): Index of the target variable in the input sequence
        (default is 0).

    Returns:
    numpy.ndarray: Reshaped target sequence with shape
        [batch_size, n_step_history, n_step_future].
    """
    y_to_concat = y[:, :, np.newaxis]
    targets_in_X = X[:, :, X_idx_target : X_idx_target + 1]
    stacked = np.concatenate((targets_in_X, y_to_concat), axis=1)

    y_ss = np.empty((stacked.shape[0], X.shape[1], y.shape[1]))
    for step_ahead in range(1, y.shape[1] + 1):
        y_ss[:, :, step_ahead - 1] = stacked[:, step_ahead : step_ahead + X.shape[1], 0]

    return y_ss


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
        """
        Initialize a Keras-based scikit-learn compatible regressor.

        Parameters:
        :param loss: str | keras.losses.Loss | None
            The loss function to be optimized during training. If None, defaults
            to Mean Squared Error.
        :param optimizer: str | keras.optimizers.Optimizer | None
            The optimizer to use during training. If None, defaults to Adam optimizer.
        :param max_epoch: int
            The maximum number of training epochs. Default is 20.
        :param patience: int
            Number of epochs with no improvement after which training will be stopped.
            Default is 2.
        :param metrics: list | None
            List of evaluation metrics to be monitored during training. Default is None.

        Attributes:
        - history: keras.callbacks.History
            The training history obtained during fitting.
        - _is_fitted: bool
            A flag indicating whether the model has been fitted or not.
        - model: keras.models.Model
            The underlying Keras model.
        """
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


class TsDeepNN(KerasModelSkBC):
    def __init__(
        self,
        loss=None,
        optimizer=None,
        max_epoch: int = 20,
        patience: int = 2,
        metrics=None,
    ):
        """
        Initialize a time-series linear model using Keras
        """
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
                keras.layers.Dense(units=X.shape[1]),
                keras.layers.Dense(units=X.shape[1]),
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


class DeepRNN(KerasModelSkBC):
    def __init__(
        self,
        n_units: int = None,
        cells: str = "LSTM",
        hidden_layers_size: int = 1,
        reshape_sequence_to_sequence: bool = True,
        loss=None,
        optimizer=None,
        max_epoch: int = 20,
        patience: int = 2,
        metrics=None,
    ):
        """
        Initialize a time-series linear model using Keras
        """
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            max_epoch=max_epoch,
            patience=patience,
            metrics=metrics,
        )

        self.cell_map = {
            "LSTM": keras.layers.LSTMCell,
            "GRU": keras.layers.GRUCell,
            "RNN": keras.layers.SimpleRNNCell,
        }

        self.cells = cells
        self.hidden_layers_size = hidden_layers_size
        self.n_units = n_units
        self.reshape_sequence_to_sequence = reshape_sequence_to_sequence

    def fit(self, X, y, x_val=None, y_val=None, idx_target: int = 0):
        if self.reshape_sequence_to_sequence:
            y = reshape_target_sequence_to_sequence(X, y, idx_target)
            if y_val is not None and x_val is not None:
                y_val = reshape_target_sequence_to_sequence(x_val, y_val, idx_target)

        model = keras.models.Sequential()
        # Input layer
        model.add(
            keras.layers.RNN(
                cell=self.cell_map[self.cells](self.n_units),
                return_sequences=True,
                input_shape=[None, X.shape[2]],
            )
        )

        # Hidden layers
        for layers in range(self.hidden_layers_size):
            model.add(
                keras.layers.RNN(
                    cell=self.cell_map[self.cells](self.n_units), return_sequences=True
                )
            )

        # Output layer
        if self.reshape_sequence_to_sequence:
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(y.shape[1])))
        else:
            model.add(keras.layers.Dense(y.shape[1]))

        model = keras.models.Sequential(
            [
                keras.layers.SimpleRNN(
                    self.n_units, return_sequences=True, input_shape=[None, X.shape[2]]
                ),
                keras.layers.SimpleRNN(self.n_units, return_sequences=True),
                keras.layers.TimeDistributed(keras.layers.Dense(y.shape[1])),
            ]
        )

        self._main_fit(model, X, y, x_val, y_val)

    def predict(self, X):
        return self.model.predict(X)
