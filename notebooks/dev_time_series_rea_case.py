# %%
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# %%
def plot_forcast(x, y_ref, model, batch_nb=42):
    fig, ax = plt.subplots()
    predictions = model.predict(x)
    if predictions.ndim == 3:
        predictions = predictions[batch_nb, -1, :]
    else:
        predictions = predictions[batch_nb, :]

    if y_ref.ndim == 3:
        y_ref_to_plot = y_ref[batch_nb, -1, :]
    else:
        y_ref_to_plot = y_ref[batch_nb, :]

    x_to_plot = x[batch_nb, :, 0]

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

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()


# %%
def last_timestep_mse(y_true, y_pred):
    return keras.metrics.mean_squared_error(y_true[:, -1], y_pred[:, -1])


# %%
def compile_and_fit(model, x, y, x_val, y_val, max_epoch=20, patience=2, metrics=None):
    if metrics is None:
        metrics = keras.metrics.MeanAbsoluteError()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(),
        metrics=[metrics],
    )

    history = model.fit(
        x,
        y,
        epochs=max_epoch,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
    )

    return history


# %%
N_STEP = 24 * 4  # 12h
N_STEP_FUTURE = 6 * 4  # 6h
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
MAX_EPOCHS = 20


# %%
if __name__ == "__main__":
    # %% DATA PREPARATION

    # COLLER LES BOUT ENSEMBLE
    # INDIQUER LE JOUR COURANT
    # INDIQUER DEMAIN

    data = pd.read_csv(
        Path(
            r"C:\Users\bdurandestebe\PycharmProjects\corrai\notebooks\ELN-CPT-ELE-GEN-TD1.1-ELNATH_15T_SINES.csv"
        )
    )

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    data = data.to_numpy()[:, 1]
    data = data.astype(np.float32)
    data = (data - data.mean()) / data.std()

    # data = np.arange(100)

    ts_data = keras.utils.timeseries_dataset_from_array(
        data=data, targets=None, sequence_length=N_STEP + N_STEP_FUTURE, shuffle=True
    )

    stacked = np.empty([0, N_STEP + N_STEP_FUTURE])
    for batch in ts_data:
        stacked = np.vstack((stacked, batch))

    stacked = stacked[..., np.newaxis]

    size = len(stacked)
    train_idx = int(size * TRAIN_RATIO)
    val_idx = int(size * (TRAIN_RATIO + VAL_RATIO))

    x_train, y_train = (
        stacked[:train_idx, :N_STEP],
        stacked[:train_idx, -N_STEP_FUTURE:, 0],
    )
    x_valid, y_valid = (
        stacked[train_idx:val_idx, :N_STEP],
        stacked[train_idx:val_idx, -N_STEP_FUTURE:, 0],
    )
    x_test, y_test = stacked[val_idx:, :N_STEP], stacked[val_idx:, -N_STEP_FUTURE:, 0]

    res_metrics = {}

    # %%
    linear = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=[N_STEP, 1]),
            keras.layers.Dense(N_STEP_FUTURE),
        ]
    )

    compile_and_fit(linear, x_train, y_train, x_valid, y_valid)

    res_metrics["linear"] = linear.evaluate(x_valid, y_valid)

    plot_forcast(x_test, y_test, linear, batch_nb=6)

    # %%
    simple_rn = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=[N_STEP, 1]),
            keras.layers.Dense(units=N_STEP),
            keras.layers.Dense(units=N_STEP),
            keras.layers.Dense(N_STEP_FUTURE),
        ]
    )

    compile_and_fit(simple_rn, x_train, y_train, x_valid, y_valid)
    res_metrics["simple_rn"] = simple_rn.evaluate(x_valid, y_valid)

    plot_forcast(x_test, y_test, simple_rn, batch_nb=6)

    # %%
    deep_rnn = keras.models.Sequential(
        [
            keras.layers.SimpleRNN(40, return_sequences=True, input_shape=[None, 1]),
            keras.layers.SimpleRNN(40),
            keras.layers.Dense(N_STEP_FUTURE),
        ]
    )

    compile_and_fit(deep_rnn, x_train, y_train, x_valid, y_valid)
    res_metrics["deep_rnn"] = deep_rnn.evaluate(x_valid, y_valid)

    # %%
    plot_forcast(x_test, y_test, deep_rnn, batch_nb=6)

    # %% New combination of y
    y = np.empty((stacked.shape[0], N_STEP, N_STEP_FUTURE))
    for step_ahead in range(1, N_STEP_FUTURE + 1):
        y[:, :, step_ahead - 1] = stacked[:, step_ahead : step_ahead + N_STEP, 0]

    y_train = y[:train_idx]
    y_valid = y[train_idx:val_idx]
    y_test = y[val_idx:]

    # %%
    improved_deep_rnn = keras.models.Sequential(
        [
            keras.layers.SimpleRNN(40, return_sequences=True, input_shape=[None, 1]),
            keras.layers.SimpleRNN(40, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(N_STEP_FUTURE)),
        ]
    )

    compile_and_fit(
        improved_deep_rnn, x_train, y_train, x_valid, y_valid, metrics=last_timestep_mse
    )
    res_metrics["improved_deep_rnn"] = improved_deep_rnn.evaluate(x_valid, y_valid)

    # %%
    plot_forcast(x_test, y_test, deep_rnn, batch_nb=6)

    # %%
    wave_net = keras.models.Sequential()
    wave_net.add(keras.layers.InputLayer(input_shape=[None, 1]))

    for rate in (1, 2, 4, 8) * 2:
        wave_net.add(
            keras.layers.Conv1D(
                filters=40,
                kernel_size=2,
                padding="causal",
                activation="relu",
                dilation_rate=rate,
            )
        )

    wave_net.add(keras.layers.Conv1D(filters=N_STEP_FUTURE, kernel_size=1))

    compile_and_fit(
        wave_net, x_train, y_train, x_valid, y_valid, metrics=last_timestep_mse
    )

    res_metrics["wave_net"] = wave_net.evaluate(x_valid, y_valid)

    # %%
    plot_forcast(x_test, y_test, wave_net, batch_nb=10)

    # %%
