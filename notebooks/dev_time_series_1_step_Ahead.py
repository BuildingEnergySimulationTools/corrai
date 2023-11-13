# %%
import keras
import numpy as np


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)


MAX_EPOCHS = 20


def compile_and_fit(model, x, y, x_val, y_val, max_epoch=20, patience=2):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        x,
        y,
        epochs=max_epoch,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
    )

    return history


if __name__ == "__main__":
    # %%
    n_steps = 50
    series = generate_time_series(10000, n_steps + 1)
    x_train, y_train = series[:7000, :n_steps], series[:7000, -1]
    x_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
    x_test, y_test = series[9000:, :n_steps], series[9000:, -1]

    res_metrics = {}

    # %%
    linear = keras.models.Sequential(
        [keras.layers.Flatten(input_shape=[50, 1]), keras.layers.Dense(1)]
    )

    compile_and_fit(linear, x_train, y_train, x_valid, y_valid)

    res_metrics["linear"] = linear.evaluate(x_valid, y_valid)

    # %%
    simple_rn = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=[50, 1]),
            keras.layers.Dense(units=50),
            keras.layers.Dense(units=50),
            keras.layers.Dense(1),
        ]
    )

    compile_and_fit(simple_rn, x_train, y_train, x_valid, y_valid)
    res_metrics["simple_rn"] = simple_rn.evaluate(x_valid, y_valid)

    # %%
    simple_rnn = keras.models.Sequential(
        [keras.layers.SimpleRNN(1, input_shape=[None, 1])]
    )

    compile_and_fit(simple_rnn, x_train, y_train, x_valid, y_valid)
    res_metrics["simple_rnn"] = simple_rnn.evaluate(x_valid, y_valid)

    # %%
    deep_rnn = keras.models.Sequential(
        [
            keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
            keras.layers.SimpleRNN(20),
            keras.layers.Dense(1),
        ]
    )

    compile_and_fit(deep_rnn, x_train, y_train, x_valid, y_valid)
    res_metrics["deep_rnn"] = deep_rnn.evaluate(x_valid, y_valid)
