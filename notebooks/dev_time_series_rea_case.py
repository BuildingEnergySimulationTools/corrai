# %%
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)


MAX_EPOCHS = 20


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
SEQUENCE_lENGTH = 10
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


# %%
if __name__ == "__main__":
    # %% DATA PREPARATION
    data = pd.read_csv(
        Path(r"C:\Users\bdurandestebe\PycharmProjects\corrai\notebooks\conso_30T.csv")
    )

    data = data.to_numpy()[:, 1]
    data = data.astype(np.float32)

    ts_data = keras.utils.timeseries_dataset_from_array(
        data=data, targets=None, sequence_length=SEQUENCE_lENGTH, shuffle=True
    )

    stacked = np.empty([0, SEQUENCE_lENGTH])
    for batch in ts_data:
        stacked = np.vstack((stacked, batch))

    size = len(stacked)
    train = stacked[0 : int(size * TRAIN_RATIO), :]
    val = stacked[int(size * TRAIN_RATIO) : int(size * (TRAIN_RATIO + VAL_RATIO)), :]
    test = stacked[int(size * (TRAIN_RATIO + VAL_RATIO)) :, :]
    #
    # # %%
    # deep_rnn = keras.models.Sequential(
    #     [
    #         keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    #         keras.layers.SimpleRNN(20),
    #         keras.layers.Dense(10),
    #     ]
    # )
    #
    # compile_and_fit(deep_rnn, x_train, y_train, x_valid, y_valid)
    # res_metrics["deep_rnn"] = deep_rnn.evaluate(x_valid, y_valid)
    #
    # # %%
    # plot_forcast(x_train, y_train, deep_rnn, batch_nb=35)
    #
    # # %% New combination of y
    # y = np.empty((10000, n_steps, 10))
    # for step_ahead in range(1, 10 + 1):
    #     y[:, :, step_ahead - 1] = series[:, step_ahead : step_ahead + n_steps, 0]
    #
    # y_train = y[:7000]
    # y_valid = y[7000:9000]
    # y_test = y[9000:]
    #
    # # %%
    # improved_deep_rnn = keras.models.Sequential(
    #     [
    #         keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    #         keras.layers.SimpleRNN(20, return_sequences=True),
    #         keras.layers.TimeDistributed(keras.layers.Dense(10)),
    #     ]
    # )
    #
    # compile_and_fit(
    #     improved_deep_rnn,
    #     x_train,
    #     y_train,
    #     x_valid,
    #     $y_valid,
    #     metrics=last_timestep_mse
    # )
    # res_metrics["improved_deep_rnn"] = improved_deep_rnn.evaluate(x_valid, y_valid)
    # # %%
    # plot_forcast(x_train, y_train, improved_deep_rnn, batch_nb=22)
    #
    # # %%
    # conv_gru = keras.models.Sequential(
    #     [
    #         keras.layers.Conv1D(
    #             filters=20,
    #             kernel_size=4,
    #             strides=2,
    #             padding="valid",
    #             input_shape=[None, 1],
    #         ),
    #         keras.layers.GRU(20, return_sequences=True),
    #         keras.layers.GRU(20, return_sequences=True),
    #         keras.layers.TimeDistributed(keras.layers.Dense(10)),
    #     ]
    # )
    #
    # # %%
    # compile_and_fit(
    #     conv_gru,
    #     x_train,
    #     y_train[:, 3::2],
    #     x_valid,
    #     y_valid[:, 3::2],
    #     metrics=last_timestep_mse,
    # )
    #
    # # %%
    # res_metrics["improved_deep_rnn"] = conv_gru.evaluate(x_valid, y_valid[:, 3::2])
    # # %%
    # plot_forcast(x_valid, y_valid[:, 3::2], conv_gru, batch_nb=22)
    #
    # # %%
    # wave_net = keras.models.Sequential()
    # wave_net.add(keras.layers.InputLayer(input_shape=[None, 1]))
    #
    # for rate in (1, 2, 4, 8) * 2:
    #     wave_net.add(
    #         keras.layers.Conv1D(
    #             filters=20,
    #             kernel_size=2,
    #             padding="causal",
    #             activation="relu",
    #             dilation_rate=rate,
    #         )
    #     )
    #
    # wave_net.add(keras.layers.Conv1D(filters=10, kernel_size=1))
    #
    # compile_and_fit(
    #     wave_net, x_train, y_train, x_valid, y_valid, metrics=last_timestep_mse
    # )
    # # %%
    # res_metrics["wave_net"] = wave_net.evaluate(x_valid, y_valid)
    #
    # # %%
    # plot_forcast(x_valid, y_valid, wave_net, batch_nb=1)
    #
    # # %%
    # wave_net.predict(x_train).shape
