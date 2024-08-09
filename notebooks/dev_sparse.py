import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from corrai.learning.time_series import plot_periodogram
from corrai.learning.model_selection import time_series_sampling
from corrai.learning.time_series import (
    sequence_prediction_to_frame,
    plot_sequence_forcast,
)
from corrai.learning.time_series import TsDeepNN, DeepRNN, SimplifiedWaveNet
from corrai.transformers import PdSkTransformer
from sklearn.preprocessing import StandardScaler
from corrai.metrics import last_time_step_mse, last_time_step_smape, smape
from sklearn.metrics import mean_absolute_error

from statsmodels.graphics.tsaplots import plot_pacf

import plotly.io as pio
import plotly.express as px

pio.renderers.default = "browser"


# %%
LABEL = ["error"]
LABEL_INDEX = 0

FEATURES = [
    "(7)ahu_multi",
    "heated_floor_ahu",
    "(3)Proj_north",
    "(10)Temperature reprise CTA RDC R+1",
    "(12)Temperature reprise CTA Multifonctionnelle",
    "(8)ahu_R1",
    "(9)Temperature soufflage CTA RDC R+1",
    "AHU_R1.Heat_to_building",
]

N_STEP_HISTORY = 24 * 2  # 24hh
N_STEP_FUTURE = 24 * 2
SAMPLING_RATE = 1
SEQUENCE_STRIDE = 1
SEED = 42

TRAIN_SIZE = 0.8
VALID_SIZE = 0.1


# %%
data = pd.read_csv(
    Path(r"C:\Users\bdurandestebe\Documents\56_NEOIA\error_ml") / "error_ml.csv",
    parse_dates=True,
    index_col=0,
)
data = data.resample("30min").mean()

# %%
abs_corr = abs(data.corr())

# %%
data = pd.concat([data[LABEL], data[FEATURES].shift(-N_STEP_FUTURE)], axis=1).dropna()

# %%
data_size = data.shape[0]
train_df = data.iloc[: int(data_size * TRAIN_SIZE), :].copy()
valid_df = data.iloc[
    int(data_size * TRAIN_SIZE) : int(data_size * (TRAIN_SIZE + VALID_SIZE)), :
].copy()
test_df = data.iloc[int(data_size * (TRAIN_SIZE + VALID_SIZE)) :, :].copy()

# %%
scaler_feat = PdSkTransformer(StandardScaler())
scaler_label = PdSkTransformer(StandardScaler())

train_df = pd.concat(
    [
        scaler_label.fit_transform(train_df[LABEL]),
        scaler_feat.fit_transform(train_df[FEATURES]),
    ],
    axis=1,
)

valid_df = pd.concat(
    [
        scaler_label.transform(valid_df[LABEL]),
        scaler_feat.transform(valid_df[FEATURES]),
    ],
    axis=1,
)

test_df = pd.concat(
    [scaler_label.transform(test_df[LABEL]), scaler_feat.transform(test_df[FEATURES])],
    axis=1,
)


# %%
plot_pacf(train_df["error"], lags=24 * 2 * 2)
plt.show()

# %%
train_np = train_df.to_numpy()
train_np = train_np.astype(np.float32)

train_sequences = time_series_sampling(
    train_np,
    sequence_length=N_STEP_HISTORY + N_STEP_FUTURE,
    sampling_rate=SAMPLING_RATE,
    sequence_stride=SEQUENCE_STRIDE,
    shuffle=True,
    seed=SEED,
)

# %%
valid_np = valid_df.to_numpy()
valid_np = valid_np.astype(np.float32)

valid_sequences = time_series_sampling(
    valid_np,
    sequence_length=N_STEP_HISTORY + N_STEP_FUTURE,
    sampling_rate=SAMPLING_RATE,
    sequence_stride=SEQUENCE_STRIDE,
    shuffle=True,
    seed=SEED,
)

# %%
x_train, y_train = (
    train_sequences[:, :N_STEP_HISTORY, :],
    train_sequences[:, -N_STEP_FUTURE:, LABEL_INDEX],
)

x_valid, y_valid = (
    valid_sequences[:, :N_STEP_HISTORY, :],
    valid_sequences[:, -N_STEP_FUTURE:, LABEL_INDEX],
)

# %%
res_metrics = {}

# %%
ts_linear = TsDeepNN(
    metrics=[smape],
    patience=200,
    max_epoch=20,
)
ts_linear.fit(x_train, y_train, x_valid, y_valid)
res_metrics["ts_linear"] = ts_linear.evaluate(x_valid, y_valid)

# %%
simple_rn = TsDeepNN(
    hidden_layers_size=3,
    metrics=[smape],
    patience=200,
    max_epoch=30,
)
simple_rn.fit(x_train, y_train, x_valid, y_valid)
res_metrics["simple_rn"] = simple_rn.evaluate(x_valid, y_valid)

# %%
# import keras
lstm_seq = DeepRNN(
    cells="LSTM",
    n_units=40,  # Reduced number of units
    hidden_layers_size=1,
    reshape_sequence_to_sequence=False,
    metrics=[smape],
    # optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Different optimizer and learning rate
    patience=2,  # Reduced patience
    max_epoch=25,  # Increased number of epochs
    loss="mse",  # Changed loss function
)
lstm_seq.fit(x_train, y_train, x_valid, y_valid)
res_metrics["lstm_seq"] = lstm_seq.evaluate(x_valid, y_valid)

# %%
gru_seq = DeepRNN(
    cells="GRU",
    n_units=40,
    hidden_layers_size=1,
    reshape_sequence_to_sequence=False,
    metrics=[smape],
    # optimizer=keras.optimizers.SGD(0.01),
    patience=2,
    max_epoch=25,
    loss="mse",
)
gru_seq.fit(x_train, y_train, x_valid, y_valid)
res_metrics["gru_seq"] = gru_seq.evaluate(x_valid, y_valid)

# %%
wave_net = SimplifiedWaveNet(
    convolutional_layers=4,
    staked_groups=4,
    groups_filters=50,
    metrics=[smape],
    patience=2,
    max_epoch=25,
    loss="mse",
)
wave_net.fit(x_train, y_train, x_valid, y_valid)
res_metrics["wave_net"] = wave_net.evaluate(x_valid, y_valid)

# %%
res_metrics

# %%
plot_sequence_forcast(x_valid, y_valid, model=simple_rn, batch_nb=5)


# %%
model_collection = {
    "ts_linear": ts_linear,
    "simple_rn": simple_rn,
    "lstm_seq": lstm_seq,
    "gru_seq": gru_seq,
    "wave_net": wave_net,
}

error_df = pd.DataFrame(
    columns=[data.index.freq * (i + 1) for i in range(N_STEP_FUTURE)],
    index=list(model_collection.keys()),
)

from corrai.metrics import nmbe

for mod_name, model in model_collection.items():
    predictions = sequence_prediction_to_frame(
        model=model,
        x_df=test_df,
        sequence_stride=1,
        sampling_rate=SAMPLING_RATE,
        n_step_history=N_STEP_HISTORY,
    )

    unscaled = pd.concat(
        [
            scaler_label.inverse_transform(predictions[col].to_frame())
            for col in predictions
        ],
        axis=1,
    )

    target_column = scaler_label.inverse_transform(test_df[LABEL])
    for col in unscaled:
        # Drop NaN values and calculate RMSE
        valid_values = pd.concat([target_column, unscaled[col]], axis=1).dropna()
        err = nmbe(valid_values[LABEL], valid_values[[col]])
        error_df.loc[mod_name, col] = err

# %%
predictions = sequence_prediction_to_frame(
    model=simple_rn,
    x_df=test_df,
    sequence_stride=1,
    n_step_history=N_STEP_HISTORY,
    sampling_rate=1,
)

# %%
unscaled = pd.concat(
    [
        scaler_label.inverse_transform(predictions[col].to_frame())
        for col in predictions
    ],
    axis=1,
)

# %%

fig = px.line(
    pd.concat([scaler_label.inverse_transform(test_df[LABEL]), unscaled], axis=1)
)
fig.show()

# %%
