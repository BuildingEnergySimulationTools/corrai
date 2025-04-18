import pandas as pd
from corrai.learning.cluster import KdeSetPoint
from corrai.learning.cluster import plot_kde_hist, plot_kde_predict

# %%
data = pd.DataFrame({"data": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2]})

# %%
kde_setpoint = KdeSetPoint(bandwidth=0.1, lik_filter=0.14)


# %%
kde_setpoint.fit_predict(data)

# %%
kde_setpoint.set_points_

# %%
plot_kde_hist(data, 0.1)

# %%
plot_kde_predict(data, estimator=kde_setpoint)

data = pd.DataFrame(
    {"data": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2]},
    index=pd.date_range("2009", periods=18, freq="h"),
)

fig = plot_kde_predict(data, bandwidth=0.1, lik_filter=0.14)
fig.show()
