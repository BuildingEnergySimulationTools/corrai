# %%
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from corrai.transformers import PdSkTransformer, PdInterpolate

from corrai.learning.model_selection import (
    time_series_sampling,
    sequences_train_test_split,
)

from corrai.transformers import PdAddFourierPairs

from corrai.learning.time_series import (
    TsDeepNN,
    DeepRNN,
    SimplifiedWaveNet,
    plot_sequence_forcast,
)

from corrai.metrics import last_time_step_rmse

# %%
sns.set(style="whitegrid")

# Set the size of the plots
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)


# %%
def plot_periodogram(ts, detrend="linear"):
    fs = pd.Timedelta("365D") / ts.index.freq
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling="spectrum",
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=freqencies,
            y=spectrum,
            mode="lines",
            line=dict(color="purple"),
        )
    )

    fig.update_layout(
        xaxis_type="log",
        xaxis=dict(
            tickvals=[1, 2, 4, 6, 12, 26, 52, 104, 365, 730, 1460, 2920, 8760, 17520],
            ticktext=[
                "Annual (1)",
                "Semiannual (2)",
                "Quarterly (4)",
                "Bimonthly (6)",
                "Monthly (12)",
                "Biweekly (26)",
                "Weekly (52)",
                "Semiweekly (104)",
                "Daily (365)",
                "12h (730)",
                "6h (1460)",
                "3h (2920)",
                "1h (8760)",
                "30min (17520)",
            ],
            tickangle=30,
        ),
        yaxis=dict(
            tickformat="e",
            title="Variance",
        ),
        title="Periodogram",
    )

    fig.show()

    return fig


# def plot_periodogram(ts, detrend="linear", ax=None):
#     fs = pd.Timedelta("365D") / ts.index.freq
#     freqencies, spectrum = periodogram(
#         ts,
#         fs=fs,
#         detrend=detrend,
#         window="boxcar",
#         scaling="spectrum",
#     )
#     if ax is None:
#         _, ax = plt.subplots()
#     ax.step(freqencies, spectrum, color="purple")
#     ax.set_xscale("log")
#     ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104, 365, 730, 1460, 2920, 8760, 17520])
#     ax.set_xticklabels(
#         [
#             "Annual (1)",
#             "Semiannual (2)",
#             "Quarterly (4)",
#             "Bimonthly (6)",
#             "Monthly (12)",
#             "Biweekly (26)",
#             "Weekly (52)",
#             "Semiweekly (104)",
#             "Daily (365)",
#             "12h (730)",
#             "6h (1460)",
#             "3h (2920)",
#             "1h (8760)",
#             "30min (17520)",
#         ],
#         rotation=30,
#     )
#     ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#     ax.set_ylabel("Variance")
#     ax.set_title("Periodogram")
#
#     plt.show()
#
#     return ax


def smape(Y_predict, Y_test):
    result = tf.norm(Y_predict - Y_test, axis=1)
    result = tf.abs(result)

    denom = tf.norm(Y_predict, axis=1) + tf.norm(Y_test, axis=1)
    result /= denom
    result *= 100 * 2

    result = tf.reduce_mean(result)
    return result


# %%
N_STEP = 12 * 4  # 12h
N_STEP_FUTURE = 6 * 4  # 6h
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
MAX_EPOCHS = 20
N_FEATURES = 4


# %%
if __name__ == "__main__":
    # %% DATA PREPARATION
    data = pd.read_csv(
        Path(
            r"C:\Users\bdurandestebe\PycharmProjects\corrai\notebooks\ELN-CPT-ELE-GEN-TD1.1-ELNATH_15T_SINES.csv"
        ),
        index_col=0,
        parse_dates=True,
    )
    data = data.asfreq("15T")
    data = data["Index (ELN-CPT-ELE-GEN-TD1.1-ELNATH)"].to_frame()

    # data_to_plot = data.fillna(0)
    # plot_periodogram(data_to_plot["Index (ELN-CPT-ELE-GEN-TD1.1-ELNATH)"])
    # plot_pacf(data_to_plot["Index (ELN-CPT-ELE-GEN-TD1.1-ELNATH)"], lags=4 * 12)
    # seasonal_decompose(
    #     data_to_plot["Index (ELN-CPT-ELE-GEN-TD1.1-ELNATH)"].to_numpy(), period=15 * 60
    # ).plot()
    # plt.show()
    # %% SET POINTS !!!!
    from corrai.learning.cluster import (
        KdeSetPointIdentificator,
        plot_time_series_kde,
        plot_kde_set_point,
    )
    from corrai.transformers import PdGaussianFilter1D
    import plotly.io as pio

    pio.renderers.default = "browser"

    # %%
    kde_pipe = make_pipeline(
        PdSkTransformer(StandardScaler()),
        PdGaussianFilter1D(sigma=4),
        KdeSetPointIdentificator(lik_filter=0.5, cluster_tol=0.2),
    )

    df = kde_pipe.fit_predict(data.dropna())
    plt.plot(df[:2000])
    plt.show()

    # %%

    # %%
    # plot_kde_set_point(data.dropna(), estimator=kde_pipe)

    # %%

    num_time_steps = data.shape[0]
    num_train, num_val = (
        int(num_time_steps * TRAIN_RATIO),
        int(num_time_steps * VAL_RATIO),
    )
    train_df = data.iloc[:num_train, :]
    train_df = train_df.loc[:"2023-05-19", :]
    val_df = data.iloc[num_train : (num_train + num_val), :]
    val_df = val_df.loc["2023-05-30":, :]
    test_df = data.iloc[(num_train + num_val) :, :]

    # %%
    def weekday_encoding(X) -> pd.DataFrame:
        X["is_working_day"] = data.index.to_series().apply(
            lambda X: 1 if X.weekday() < 5 else 0
        )
        return X

    # %%
    def is_people(X) -> pd.DataFrame:
        X["is_people"] = kde_pipe.fit_predict(X.dropna())
        X.loc[X["is_people"] == -1, "is_people"] = 1
        return X

    # %%

    # pre_process_pipe = make_pipeline(
    #     PdSkTransformer(StandardScaler()),
    #     PdInterpolate(method="linear"),
    #     # PdSkTransformer(FunctionTransformer(func=is_people)),
    #     PdSkTransformer(FunctionTransformer(func=weekday_encoding)),
    #     PdAddFourierPairs(
    #         frequency=1 / (7 * 24 * 3600), feature_prefix="week", order=2
    #     ),
    # )
    pre_process_pipe = make_pipeline(
        PdSkTransformer(StandardScaler()),
        PdInterpolate(method="linear"),
        PdSkTransformer(FunctionTransformer(func=is_people)),
        PdSkTransformer(FunctionTransformer(func=weekday_encoding)),
        PdAddFourierPairs(frequency=1 / (7 * 24 * 3600), feature_prefix="week"),
        PdAddFourierPairs(frequency=1 / (3.5 * 24 * 3600), feature_prefix="1/2week"),
        PdAddFourierPairs(frequency=1 / (1 * 24 * 3600), feature_prefix="day"),
        PdAddFourierPairs(frequency=1 / (6 * 3600), feature_prefix="6h"),
        PdAddFourierPairs(frequency=1 / (3 * 3600), feature_prefix="3h")
        # PdAddFourierPairs(frequency=1 / (2 * 3600)),
        # PdAddFourierPairs(frequency=1 / (1 * 3600)),
    )
    test = pre_process_pipe.fit_transform(
        data["Index (ELN-CPT-ELE-GEN-TD1.1-ELNATH)"].to_frame()
    )

    test.iloc[:1000, :3].plot()
    plt.show()
    corr = abs(test.corr())
    # data["sum"] = data.iloc[:, 1:].sum(axis=1)
    # # data.loc["2023-02", ["Index (ELN-CPT-ELE-GEN-TD1.1-ELNATH)", "sum"]].plot()
    # data.loc["2023-02", "is_people"].plot()
    # plt.show()

    # %%
    train_df = pre_process_pipe.fit_transform(train_df)
    train_array = train_df.to_numpy()
    train_array = train_array.astype(np.float32)

    val_df = pre_process_pipe.transform(val_df)
    val_array = val_df.to_numpy()
    val_array = val_array.astype(np.float32)

    test_df = pre_process_pipe.transform(test_df)
    test_array = test_df.to_numpy()
    test_array = test_array.astype(np.float32)

    print(np.any(np.isnan(train_array)))
    print(np.any(np.isnan(val_array)))
    print(np.any(np.isnan(test_array)))

    # %%
    res_metrics = {}

    # %% TRAINING
    train = time_series_sampling(
        train_array,
        sequence_length=N_STEP + N_STEP_FUTURE,
        sampling_rate=1,
        sequence_stride=1,
        shuffle=False,
        seed=42,  # Make sure the behaviour can be repeated in the notebook
    )

    x_train, y_train = (
        train[:, :N_STEP, :],
        train[:, -N_STEP_FUTURE:, 0],
    )

    # %%
    valid = time_series_sampling(
        val_array,
        sequence_length=N_STEP + N_STEP_FUTURE,
        sampling_rate=1,
        sequence_stride=1,
        shuffle=False,
        seed=42,  # Make sure the behaviour can be repeated in the notebook
    )

    x_valid, y_valid = (
        valid[:, :N_STEP, :],
        valid[:, -N_STEP_FUTURE:, 0],
    )

    # %% TRAINING
    test = time_series_sampling(
        test_array,
        sequence_length=N_STEP + N_STEP_FUTURE,
        sampling_rate=1,
        sequence_stride=1,
        shuffle=False,
        seed=42,  # Make sure the behaviour can be repeated in the notebook
    )

    x_test, y_test = (
        test[:, :N_STEP, :],
        test[:, -N_STEP_FUTURE:, 0],
    )

    # %%
    ts_linear = TsDeepNN()
    ts_linear.fit(x_train, y_train, x_valid, y_valid)
    res_metrics["ts_linear"] = ts_linear.evaluate(x_valid, y_valid)
    plot_sequence_forcast(x_valid, y_valid, model=ts_linear, batch_nb=200)

    # %%
    simple_rn = TsDeepNN(hidden_layers_size=3)
    simple_rn.fit(x_train, y_train, x_valid, y_valid)
    res_metrics["simple_rn"] = simple_rn.evaluate(x_valid, y_valid)
    plot_sequence_forcast(x_valid, y_valid, model=simple_rn, batch_nb=200)
    # %%
    lstm_seq = DeepRNN(
        cells="LSTM",
        n_units=40,
        hidden_layers_size=1,
        reshape_sequence_to_sequence=True,
        metrics=[last_time_step_rmse],
        # optimizer=keras.optimizers.SGD(0.01),
        # patience=200,
        max_epoch=20,
        # loss=smape,
    )

    lstm_seq.fit(x_train, y_train, x_valid, y_valid)
    res_metrics["lstm_seq"] = lstm_seq.evaluate(x_valid, y_valid)

    # %%
    plot_sequence_forcast(x_test, y_test, model=lstm_seq, batch_nb=250)

    # %%
    gru_seq = DeepRNN(
        cells="GRU",
        n_units=40,
        hidden_layers_size=1,
        reshape_sequence_to_sequence=True,
        metrics=[last_time_step_rmse],
        # optimizer=keras.optimizers.SGD(0.01),
        # patience=200,
        # max_epoch=25,
        # loss=smape,
    )
    gru_seq.fit(x_train, y_train, x_valid, y_valid)
    res_metrics["gru_seq"] = gru_seq.evaluate(x_valid, y_valid)
    # %%
    plot_sequence_forcast(x_train, y_train, model=gru_seq, batch_nb=250)
    # %%
    plot_sequence_forcast(x_test, y_test, model=gru_seq, batch_nb=150)

    # %%
    wave_net = SimplifiedWaveNet(
        convolutional_layers=4,
        staked_groups=2,
        groups_filters=50,
        metrics=[last_time_step_rmse],
        # optimizer=keras.optimizers.SGD(0.01),
        # patience=200,
        max_epoch=25,
        # loss=smape,
    )
    wave_net.fit(x_train, y_train, x_valid, y_valid)
    res_metrics["wave_net"] = wave_net.evaluate(x_valid, y_valid)
    # %%
    plot_sequence_forcast(x_test, y_test, model=wave_net, batch_nb=300)

    # %%
    # from corrai.learning.time_series import reshape_target_sequence_to_sequence
    #
    # y_train_reshaped = reshape_target_sequence_to_sequence(x_train, y_train)
    # y_val_reshaped = reshape_target_sequence_to_sequence(x_valid, y_valid)
    #
    # # %%
    # model = keras.models.Sequential(
    #     [
    #         keras.layers.Conv1D(
    #             filters=10,
    #             kernel_size=4,
    #             strides=1,
    #             padding="valid",
    #             input_shape=[None, 16],
    #         ),
    #         keras.layers.GRU(40, return_sequences=True),
    #         keras.layers.GRU(40, return_sequences=True),
    #         keras.layers.TimeDistributed(keras.layers.Dense(24)),
    #     ]
    # )
    #
    # model.compile(loss=smape, optimizer="adam", metrics=[last_time_step_rmse])
    # model.fit(
    #     x_train,
    #     y_train_reshaped[:, 3::1],
    #     epochs=20,
    #     validation_data=(x_valid, y_val_reshaped[:, 3::1]),
    # )
    #
    # # %%
    # plot_sequence_forcast(x_valid, y_val_reshaped, model=model, batch_nb=100)
    # %%

    predictions = pd.DataFrame(lstm_seq.predict(x_valid))
    predictions.index = pd.date_range(
        val_df.index.to_series().iloc[N_STEP + 1],
        freq="15T",
        periods=predictions.shape[0],
    )

    concat = pd.concat(
        [val_df["Index (ELN-CPT-ELE-GEN-TD1.1-ELNATH)"]]
        + [predictions[col].shift(col - 1) for col in predictions],
        axis=1,
    )

    # %%
    concat.iloc[:500, [0, 23]].plot()
    plt.show()

    # %%
    residu_23 = concat.iloc[:, 0] - concat.iloc[:, 23]
    # plt.plot(residu_23[:500])
    # plt.plot(concat.iloc[:500, [0, 24]])
    # fig = plot_periodogram(residu_23.dropna())
    plt.hist(residu_23.dropna())
    plt.show()

    # %%
    from statsmodels.graphics.tsaplots import plot_acf

    plot_acf(residu_23.dropna())
    plt.show()

    # %%
    # Work with residue

    residu = pd.DataFrame(y_train - gru_seq.predict(x_train))
    residu.index = pd.date_range(
        val_df.index.to_series().iloc[N_STEP + 1],
        freq="15T",
        periods=residu.shape[0],
    )

    concat = pd.concat(
        [val_df] + [residu[col].shift(col - 1) for col in residu],
        axis=1,
    )

    # %%
    corr = concat.corr()

    # %%

    # %%
    y_residu = y_train - gru_seq.predict(x_train)
    y_residu_train = y_residu[: int(y_residu.shape[0] * 0.8), :]
    y_residu_val = y_residu[int(y_residu.shape[0] * 0.8) :, :]
    model_residu = DeepRNN(cells="GRU", n_units=40, reshape_sequence_to_sequence=False)
    model_residu.fit(
        x_train[: int(y_residu.shape[0] * 0.8), :, [6, 10, 11]],
        y_residu_train,
        x_train[int(y_residu.shape[0] * 0.8) :, :, [6, 10, 11]],
        y_residu_val,
    )

    # %%
    plot_sequence_forcast(
        x_train[:, :, [6, 10, 11]], y_residu, model=model_residu, batch_nb=300
    )

    # # %% STACK MODELS
    # class TwoStackedModels:
    #     def __init__(self, first_model, second_model):
    #         self.first_model = first_model
    #         self.second_model = second_model
    #
    #     def fit(
    #         self,
    #         X,
    #         y,
    #         x_valid,
    #         y_valid,
    #         residu_n_step_history=6,
    #         residu_train_split=0.8,
    #     ):
    #         self.first_model.fit(X, y, x_valid, y_valid)
    #
    #         # Work with residu
    #         y_resid = y - self.first_model.predict(X)
    #
    #         residu_train = y_resid[: int(y_resid.shape[0] * residu_train_split), :]
    #         residu_val = y_resid[int(y_resid.shape[0] * residu_train_split) :, :]
    #
    #         train_sampled_residue = time_series_sampling(
    #             residu_train, sequence_length=1 + residu_n_step_history
    #         )
    #         x_residu_train, y_residu_train = (
    #             train_sampled_residue[:, :residu_n_step_history, :],
    #             train_sampled_residue[:, -1:, :],
    #         )
    #
    #         y_residu_train = np.squeeze(y_residu_train, axis=1)
    #
    #         # %%
    #         val_sampled_residue = time_series_sampling(
    #             residu_val, sequence_length=1 + residu_n_step_history
    #         )
    #         x_residu_val, y_residu_val = (
    #             val_sampled_residue[:, :residu_n_step_history, :],
    #             val_sampled_residue[:, -1:, :],
    #         )
    #
    #         y_residu_val = np.squeeze(y_residu_val, axis=1)
    #
    #         self.second_model.fit(
    #             x_residu_train, y_residu_train, x_residu_val, y_residu_val
    #         )
    #
    #     def predict(self, X):
    #         self.first_model.predict(X)
    #
    #         self.second_model.predict(X)
    #
    # # %%
    # big = TwoStackedModels(
    #     first_model=DeepRNN(
    #         cells="GRU",
    #         n_units=40,
    #         hidden_layers_size=1,
    #         reshape_sequence_to_sequence=True,
    #         metrics=[last_time_step_rmse],
    #     ),
    #     second_model=SimplifiedWaveNet(
    #         convolutional_layers=4,
    #         staked_groups=2,
    #         groups_filters=50,
    #         metrics=[last_time_step_rmse],
    #     ),
    # )
    #
    # big.fit(x_train, y_train, x_valid, y_valid)
    # # %%
    # np.mean(mean_squared_error(y_valid, big.predict(x_valid)))
    # # %%
    # plot_sequence_forcast(x_valid, y_valid, model=big, batch_nb=250)
    #
    # # %%
    # residu_model = SimplifiedWaveNet()
    # residu_model.fit(x_train, y_resid)
    #
    # # %%
    # full_pred = gru_seq.predict(x_valid) + residu_model.predict(x_valid)
    #
    # # %%
    # predictions = pd.DataFrame(full_pred)
    # predictions.index = pd.date_range(
    #     val_df.index.to_series().iloc[N_STEP + 1],
    #     freq="15T",
    #     periods=predictions.shape[0],
    # )
    #
    # concat = pd.concat(
    #     [val_df["Index (ELN-CPT-ELE-GEN-TD1.1-ELNATH)"]]
    #     + [predictions[col].shift(col - 1) for col in predictions],
    #     axis=1,
    # )
    #
    # # %%
    # concat.iloc[500:, [0, 2]].plot()
    # plt.show()
    #
    # # %%
    # plot_pacf(y_resid[:, 23])
    # plt.plot(y_resid[:, 23])
    # plt.show()
    #
    # # %% TUNNING
    # from sklearn.metrics import mean_squared_error
    #
    # grid = GridSearchCV(
    #     gru_seq,
    #     param_grid={
    #         "n_units": [20, 40, 60],
    #         "hidden_layers_size": [1, 2, 4],
    #         "reshape_sequence_to_sequence": [True, False],
    #     },
    #     scoring=mean_squared_error,
    # )
    #
    # grid.fit(x_train, y_train)
    #
    # # %%
    # import statsmodels.api as sm
    # from matplotlib import rcParams
    # import datetime as dt
    #
    # data = pd.read_csv(
    #     Path(
    #         r"C:\Users\bdurandestebe\PycharmProjects\corrai\notebooks\ELN-CPT-ELE-GEN-TD1.1-ELNATH_15T_SINES.csv"
    #     ),
    #     index_col=0,
    # )
    #
    # data.index = pd.to_datetime(data.index)
    # data = data.resample(dt.timedelta(minutes=15)).mean()
    # data = data.interpolate("linear")
    # # %%
    # decomposition = sm.tsa.seasonal_decompose(
    #     data.iloc[:, 0], period=dt.timedelta(minutes=15)
    # )
    #
    # rcParams["figure.figsize"] = 16, 4
    # decomposition.seasonal.plot()
