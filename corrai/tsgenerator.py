import pandas as pd
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

coefficients_COSTIC = {
    "month": {
        "January": 1.11,
        "February": 1.20,
        "March": 1.11,
        "April": 1.06,
        "May": 1.03,
        "June": 0.93,
        "July": 0.84,
        "August": 0.72,
        "September": 0.92,
        "October": 1.03,
        "November": 1.04,
        "December": 1.01,
    },
    "week": {
        "Monday": 0.97,
        "Tuesday": 0.95,
        "Wednesday": 1.00,
        "Thursday": 0.97,
        "Friday": 0.96,
        "Saturday": 1.02,
        "Sunday": 1.13,
    },
    "day": {
        "hour_weekday": np.array(
            [
                0.264,
                0.096,
                0.048,
                0.024,
                0.144,
                0.384,
                1.152,
                2.064,
                1.176,
                1.08,
                1.248,
                1.224,
                1.296,
                1.104,
                0.84,
                0.768,
                0.768,
                1.104,
                1.632,
                2.088,
                2.232,
                1.608,
                1.032,
                0.624,
            ]
        ),
        "hour_saturday": np.array(
            [
                0.408,
                0.192,
                0.072,
                0.048,
                0.072,
                0.168,
                0.312,
                0.624,
                1.08,
                1.584,
                1.872,
                1.992,
                1.92,
                1.704,
                1.536,
                1.2,
                1.248,
                1.128,
                1.296,
                1.32,
                1.392,
                1.2,
                0.936,
                0.696,
            ]
        ),
        "hour_sunday": np.array(
            [
                0.384,
                0.168,
                0.096,
                0.048,
                0.048,
                0.048,
                0.12,
                0.216,
                0.576,
                1.128,
                1.536,
                1.752,
                1.896,
                1.872,
                1.656,
                1.296,
                1.272,
                1.248,
                1.776,
                2.016,
                2.04,
                1.392,
                0.864,
                0.552,
            ]
        ),
    },
}
coefficients_RE2020 = {
    "month": {
        "January": 1.05,
        "February": 1.05,
        "March": 1.05,
        "April": 0.95,
        "May": 0.95,
        "June": 0.95,
        "July": 0.95,
        "August": 0.95,
        "September": 0.95,
        "October": 1.05,
        "November": 1.05,
        "December": 1.05,
    },
    "day": {
        "hour_weekday": np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.028,
                0.029,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.007,
                0.022,
                0.022,
                0.022,
                0.007,
                0.007,
                0.007,
            ]
        ),
        "hour_weekend": np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.028,
                0.029,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.011,
                0.011,
                0.029,
                0.011,
                0.0011,
                0.0011,
                0.0,
            ]
        ),
    },
}


class DHWaterConsumption:
    """
    Class for calculating domestic hot water consumption in a building based on either
    RE2020 or COSTIC coefficients.

    Parameters:
    -----------
    n_dwellings : int
        Number of dwellings in the building.
    v_per_dwelling : float, optional (default=110)
        Daily hot water consumption per dwelling in liters.
    ratio_bath_shower : float, optional (default=0.8)
        Ratio of baths compared to showers.
    t_shower : int, optional (default=7)
        Average duration of a shower in minutes.
    d_shower : float, optional (default=8)
       Shower flow rate of showers in liters per minute.
    s_moy_dwelling : float, optional (default=49.6)
        Average surface area per dwelling in square meters.
    s_tot_building : float, optional (default=2480)
        Total surface area of the building in square meters.
    method : str, optional (default="COSTIC")
        Method for calculating hot water consumption. Available options are "COSTIC" or
         "RE2020".

    Attributes:
    -----------
    df_coefficient : pd.DataFrame
        DataFrame containing the  coefficients for hot water consumption, calculated
        either on "COSTIC" or "RE2020" method.
    df_daily_sum : pd.DataFrame
        DataFrame containing the daily sum of coefficients.
    df_re2020 : pd.DataFrame
        DataFrame containing the hot water consumption for showers based on "RE2020"
        method.
    df_costic : pd.DataFrame
        DataFrame containing the hot water consumption for showers based on "COSTIC"
        method.
    df_costic_random : pd.DataFrame
        DataFrame containing a random distribution of hot water consumption for showers
        based on "COSTIC" method.
    df_all : pd.DataFrame
        DataFrame containing calculated coefficients and other calculated data.
    coefficients : dict
        Dictionary containing the coefficients for calculating hot water consumption
        based on the chosen method.
    v_used : float
        Total hot water consumption per shower in liters.
    v_liters_day : float
        Total hot water consumption per day for the entire building in liters.
    v_shower_bath_per_day : float
        Total hot water consumption per day for the entire building in liters taking
        into account the ratio_bath_shower.

    Methods:
    --------
    get_coefficient_calc_from_period(start, end):
        Calculates the coefficients for hot water consumption based on the chosen
        method within the given time period.
    costic_shower_distribution(start, end):
        Calculates the hot water consumption for showers based on "COSTIC" method
        within the given time period. The output dataframe is sampled every hour.
    costic_random_shower_distribution(start, end, optional_columns=False, seed=None):
        Calculates a random distribution of hot water consumption for showers based on
        "COSTIC" method within  the given time period. The output dataframe is sampled
        every minute.
        Set seed to None for a new random distribution each time method is run.
        Otherwise, set to integer for the same random distribution.
    re2020_shower_distribution(start, end):
        Calculates the hot water consumption for showers based on "COSTIC" method
        within the given time period. The output dataframe is sampled every hour.
    """

    def __init__(
        self,
        n_dwellings,
        v_per_dwelling=110,
        ratio_bath_shower=0.8,
        t_shower=7,
        d_shower=8,
        s_moy_dwelling=49.6,
        s_tot_building=2480,
        method="COSTIC",
    ):
        self.method = method
        self.n_dwellings = n_dwellings
        self.v_per_dwelling = v_per_dwelling
        self.ratio_bath_shower = ratio_bath_shower
        self.t_shower = t_shower
        self.d_shower = d_shower
        self.s_moy_dwelling = s_moy_dwelling
        self.s_tot_building = s_tot_building

        self.df_coefficient = None
        self.df_daily_sum = None
        self.df_re2020 = None
        self.df_costic = None
        self.df_costic_random = None
        self.df_all = None

        if self.method == "COSTIC":
            self.coefficients = coefficients_COSTIC
        elif self.method == "RE2020":
            self.coefficients = coefficients_RE2020

        self.v_used = self.t_shower * self.d_shower
        self.v_liters_day = self.n_dwellings * self.v_per_dwelling
        self.v_shower_bath_per_day = self.ratio_bath_shower * self.v_liters_day

    def get_coefficient_calc_from_period(self, start, end):
        if not pd.Timestamp(start) or not pd.Timestamp(end):
            raise ValueError("Start and end values must be valid timestamps.")

        date_index = pd.date_range(start=start, end=end, freq="H")
        self.df_coefficient = pd.DataFrame(
            data=np.zeros(date_index.shape[0]), index=date_index, columns=["coef"]
        )

        val_list = []
        if self.method == "COSTIC":
            for val in self.df_coefficient.index:
                if val.day_of_week in range(5):
                    hour_coefficients = self.coefficients["day"]["hour_weekday"]
                elif val.day_of_week == 5:
                    hour_coefficients = self.coefficients["day"]["hour_saturday"]
                else:
                    hour_coefficients = self.coefficients["day"]["hour_sunday"]

                h24 = (
                    hour_coefficients
                    * self.coefficients["month"][str(val.month_name())]
                    * self.coefficients["week"][str(val.day_name())]
                )
                val_list.append(h24[val.hour])

        elif self.method == "RE2020":
            for val in self.df_coefficient.index:
                if val.day_of_week in range(5):
                    hour_coefficients = self.coefficients["day"]["hour_weekday"]
                else:
                    hour_coefficients = self.coefficients["day"]["hour_weekend"]

                h24 = (
                    hour_coefficients
                    * self.coefficients["month"][str(val.month_name())]
                )
                val_list.append(h24[val.hour])

        self.df_coefficient["coef"] = val_list
        self.df_daily_sum = self.df_coefficient.resample("D").sum()
        self.df_daily_sum.columns = ["coef_daily_sum"]
        return self.df_coefficient

    def costic_shower_distribution(self, start, end):
        # Concatenation of coefficients and their daily sum
        self.get_coefficient_calc_from_period(start, end)
        df_co = pd.concat([self.df_coefficient, self.df_daily_sum], axis=1)
        df_co.fillna(method="ffill", inplace=True)
        df_co = df_co.dropna(axis=0)

        # Calculation of the number of showers per hour
        df_co["consoECS_COSTIC"] = (
            df_co["coef"] * self.v_shower_bath_per_day / df_co["coef_daily_sum"]
        )
        return df_co[["consoECS_COSTIC"]]

    def costic_random_shower_distribution(
        self, start=None, end=None, optional_columns=False, seed=None
    ):
        if seed is not None:
            rs = RandomState(MT19937(SeedSequence(seed)))
        else:
            rs = RandomState()

        periods = pd.date_range(start=start, end=end, freq="T")
        df_costic = self.costic_shower_distribution(start, end)
        df_costic["nb_shower"] = df_costic["consoECS_COSTIC"] / self.v_used
        df_costic["t_shower_per_hour"] = df_costic["nb_shower"] * self.t_shower
        df_costic["nb_shower_int"] = np.round(df_costic["nb_shower"]).astype(int)

        rs_dd = rs.randint(
            0, 60 - self.t_shower, (len(periods), df_costic["nb_shower_int"].max())
        )

        distribution_list = []
        for h, nb_shower in zip(rs_dd, df_costic["nb_shower_int"]):
            starts = h[:nb_shower]
            distribution = np.zeros(60)
            for start_shower in starts:
                distribution[start_shower : start_shower + self.t_shower] += 1
            distribution_list.append(distribution)

        df_costic_random = pd.DataFrame(
            data=np.concatenate(distribution_list),
            index=pd.date_range(
                df_costic["nb_shower_int"].index[0],
                freq="T",
                periods=df_costic.shape[0] * 60,
            ),
            columns=["shower_per_minute"],
        )

        df_costic_random["consoECS_COSTIC_random"] = (
            df_costic_random["shower_per_minute"] * self.v_used / self.t_shower
        )
        df_costic_random["consoECS_COSTIC_random"] = df_costic_random[
            "consoECS_COSTIC_random"
        ].astype(float)

        self.df_costic_random = df_costic_random

        if optional_columns:
            return df_costic_random
        else:
            return df_costic_random[["consoECS_COSTIC_random"]]

    def re2020_shower_distribution(self, start, end):
        periods = pd.date_range(start=start, end=end, freq="H")
        self.get_coefficient_calc_from_period(start, end)
        # N_calculation
        if self.s_moy_dwelling < 10:
            nmax = 1
        elif self.s_moy_dwelling in {10: 49.99}:
            nmax = 1.75 - 0.01875 * (50 - self.s_moy_dwelling)
        else:
            nmax = 0.035 * self.s_moy_dwelling

        n_adult = nmax * self.n_dwellings
        a = min(392, int(40 * (self.s_tot_building / self.s_moy_dwelling)))
        v_weekly = a * n_adult  # Liters
        v_shower_bath = v_weekly * self.ratio_bath_shower

        # Calculation of the number of showers per hour
        to_return = self.df_coefficient.copy() * v_shower_bath
        to_return = to_return[: len(periods)]
        to_return.columns = ["consoECS_RE2020"]
        return to_return


class Scheduler:
    def __init__(self, config_dict=None):
        self.config_dict = config_dict

    def _get_day_dict(self):
        day_dict = {}
        for day in self.config_dict["DAYS"].keys():
            day_df = pd.DataFrame(index=["00:00"])
            for hour, prog in self.config_dict["DAYS"][day].items():
                day_df.loc[hour, prog.keys()] = prog.values()
            day_dict[day] = day_df

        return day_dict

    def get_dataframe(self, freq="T"):
        year = self.config_dict["YEAR"]

        day_list = []
        for period in self.config_dict["PERIODS"]:
            period_index = pd.date_range(
                start=f"{year}-{period[0][0]}",
                end=f"{year}-{period[0][1]}",
                freq="D",
                tz=self.config_dict["TZ"],
            )
            day_dict = self._get_day_dict()

            week = self.config_dict["WEEKS"][period[1]]
            for day in week.keys():
                ref_day = day_dict[week[day]]
                day_index = period_index[period_index.day_name() == day]
                for d in day_index:
                    current_day = ref_day.copy()
                    date = d.date()
                    new_index = pd.to_datetime(
                        date.strftime("%Y-%m-%d ") + current_day.index
                    )
                    current_day.index = new_index
                    day_list.append(current_day)

        df = pd.concat(day_list)
        df = df.fillna(method="bfill")
        df = df.resample("T").bfill()
        df = df.shift(-1)
        df = df.fillna(method="ffill")
        return df.resample(freq).mean()
