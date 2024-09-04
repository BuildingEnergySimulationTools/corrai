from pathlib import Path
import datetime as dt
import pandas as pd
import numpy as np

from statsmodels.graphics.tsaplots import plot_pacf

from corrai.fmu import ModelicaFmuModel
from corrai.measure import MeasuredDats, Transformer
from corrai.transformers import (
    PdSkTransformer,
    PdAddTimeLag,
    PdDropColumns,
    PdRenameColumns,
    PdAddFourierPairs
)
from corrai.learning.time_series import (
    time_series_sampling,
    TsDeepNN,
    plot_sequence_forcast,
    sequence_prediction_to_frame,
)
from corrai.metrics import smape, nmbe, cv_rmse
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt

#%%
pio.renderers.default = "browser"


 # %%
def split_x_y_sequence(
    df,
    n_step_history,
    n_step_future,
    sampling_rate,
    sequence_stride,
    label_index,
    shuffle=True,
    seed=42,
):
    res_np = df.to_numpy()
    res_np = res_np.astype(np.float32)

    sequences = time_series_sampling(
        res_np,
        sequence_length=n_step_history + n_step_future,
        sampling_rate=sampling_rate,
        sequence_stride=sequence_stride,
        shuffle=shuffle,
        seed=seed,
    )

    return (
        sequences[:, :n_step_history, :],
        sequences[:, -n_step_future:, label_index],
    )

# %%
MODEL_RESULT = "heating_Cooling_SP.Control_out"
REFERENCE = "Boundaries.y[6]"
CORRECTED_NAME = "corrected"
FEATURES = [
    "(7)ahu_multi",
    "heated_floor_ahu",
    # "(3)Proj_north",
    # "(10)Temperature reprise CTA RDC R+1",
    "(12)Temperature reprise CTA Multifonctionnelle",
    "(8)ahu_R1",
    # "(9)Temperature soufflage CTA RDC R+1",
    "AHU_R1.Heat_to_building",
]

N_STEP_HISTORY = 24 * 2  # 24hh
N_STEP_FUTURE = 24 * 2
SAMPLING_RATE = 1
SEQUENCE_STRIDE = 1
SEED = 42

TRAIN_SIZE = 0.8
VALID_SIZE = 0.1

LABEL = "error"
LABEL_INDEX = 0

# %%
data = pd.read_csv(
    Path(r"C:\Users\bdurandestebe\Documents\56_NEOIA\error_ml") / "error_ml.csv",
    parse_dates=True,
    index_col=0,
)
data = data.resample("30min").mean()

data = data[[LABEL] + FEATURES]

data_size = data.shape[0]
train_df = data.iloc[: int(data_size * TRAIN_SIZE), :].copy()
valid_df = data.iloc[
    int(data_size * TRAIN_SIZE) : int(data_size * (TRAIN_SIZE + VALID_SIZE)), :
].copy()
test_df = data.iloc[int(data_size * (TRAIN_SIZE + VALID_SIZE)) :, :].copy()


# %%
def error(X):
    X = X.copy()
    X[LABEL] = X[REFERENCE] - X[MODEL_RESULT]
    return X


process_results_pipe = Pipeline(
    [
        # ("compute_error", FunctionTransformer(error)),
        (
            "shift_error",
            PdAddTimeLag(
                features_to_lag=LABEL,
                time_lag=dt.timedelta(hours=N_STEP_HISTORY / 2),
                drop_resulting_nan=True,
                feature_marker="marked_",
            ),
        ),
        (
            "drop_original_error",
            PdDropColumns(to_drop=[LABEL]),
        ),
        ("rename", PdRenameColumns(new_names={f"marked_{LABEL}": LABEL})),
        ("6h_fp", PdAddFourierPairs(frequency=1 / (6 * 3600), feature_prefix="6h")),
        ("12h_fp", PdAddFourierPairs(frequency=1 / (12 * 3600), feature_prefix="12h")),
        ("24h_fp", PdAddFourierPairs(frequency=1 / (24 * 3600), feature_prefix="24h")),
        (
            "label_features_wise_scaler",
            ColumnTransformer(
                [
                    ("label_scaler", PdSkTransformer(StandardScaler()), [LABEL]),
                ],
                remainder=PdSkTransformer(StandardScaler()),
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
    ]
)

# %%
train_df = process_results_pipe.fit_transform(train_df)

x_train, y_train = split_x_y_sequence(
    df=train_df,
    n_step_history=N_STEP_HISTORY,
    n_step_future=N_STEP_FUTURE,
    sampling_rate=SAMPLING_RATE,
    sequence_stride=SEQUENCE_STRIDE,
    label_index=LABEL_INDEX,
)

# %%
valid_df = process_results_pipe.transform(valid_df)

x_valid, y_valid = split_x_y_sequence(
    df=valid_df,
    n_step_history=N_STEP_HISTORY,
    n_step_future=N_STEP_FUTURE,
    sampling_rate=SAMPLING_RATE,
    sequence_stride=SEQUENCE_STRIDE,
    label_index=LABEL_INDEX,
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
corr = abs(data.corr())

#%%

plot_sequence_forcast(x_valid, y_valid, model=simple_rn, batch_nb=20)

# %%
test_df = process_results_pipe.transform(test_df)

# %%
pred_test = sequence_prediction_to_frame(
    model=simple_rn,
    x_df=test_df,
    n_step_history=N_STEP_HISTORY,
    sampling_rate=SAMPLING_RATE,
    sequence_stride=SEQUENCE_STRIDE,
)

# %%
column_transformer = process_results_pipe.named_steps["label_features_wise_scaler"]
label_scaler = column_transformer.named_transformers_["label_scaler"]

unscaled = pd.concat(
    [label_scaler.inverse_transform(pred_test[col].to_frame()) for col in pred_test],
    axis=1,
)

# %%
fig = px.line(
    pd.concat([label_scaler.inverse_transform(test_df[[LABEL]]), unscaled], axis=1)
)
fig.show()


# %%
