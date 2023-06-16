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
        """
        Scheduler class for generating a schedules in a DataFrame object based on
        configuration settings.

        :param dict config_dict: A dictionary containing the configuration settings
            for the scheduler. See bellow for a dictionary example.

        Guidelines for config_dict:
            - The schedules hour means "until". The first specified hour correspond to
            the first switch. In the example below, heating is 17 until 09:15 when it
            becomes 21.
            - If the set-point is constant through the day, provide the last hour of
            the day ("23:00") and the corresponding value.
            - The Scheduler allows the specification of several schedules at once.
            Each key provided in the hour dictionary correspond to a new column. If
            a key is absent at a specific hour, the previous set-point value is
            propagated

        Example of config_dict structure:
            {
                "DAYS": {
                    "working_day": {
                        "09:15": {"heating": 17, "extraction_flow_rate": 0},
                        "18:00": {"heating": 21, "extraction_flow_rate": 3000},
                        "19:00": {"heating": 22},
                        "23:00": {"heating": 17, "extraction_flow_rate": 0},
                    },
                    "Off": {
                        "23:00": {"heating": 0, "extraction_flow_rate": 0},
                    },
                    ...
                },
                "WEEKS": {
                    "winter_week": {
                        "Monday": "working_day",
                        ...
                    },
                    "summer_week": {
                        "Monday": "Off",
                        ...
                    },
                    ...
                },
                "PERIODS": [
                    (("2009-01-01", "2009-03-31"), "winter_week"),
                    (("2009-04-01", "2009-09-30"), "summer_week"),
                    (("2009-10-01", "2009-12-31"), "winter_week"),
                ],
                "TZ": "Europe/Paris",
            }

        Guidelines for config_dict:
            - The "DAYS" key should contain a dictionary where the keys represent the
            names of the schedule. Each schedule is a dictionary with hourly settings.

            - The hour in the schedule dictionary represents the "until" time.
            The first specified hour corresponds to the first switch. In the example,
            "heating" is 17 until 09:15 when it becomes 21.

            - If a set-point is constant throughout the day, provide the last hour of
            the day ("23:00") and the corresponding value.

            - The Scheduler allows the specification of several schedules at once.
            Each key provided in the hour dictionary corresponds to a new column.
            If a key is absent at a specific hour, the previous set-point value is
            propagated.

        Note:
            The config_dict should be provided during initialization and should have
            the following structure:
            - "DAYS": A dictionary containing daily schedules with specific timings
            and values.

            - "WEEKS": A dictionary mapping weekdays to the corresponding daily
            schedules.

            - "PERIODS": A list of tuples representing date periods and corresponding
            week schedules.

            - "TZ": The timezone for the scheduler.

        Example:
            scheduler = Scheduler(config_dict)
            df = scheduler.get_dataframe()

        """

        self.config_dict = config_dict

    def _get_day_dict(self):
        """
        Returns a dictionary with daily schedules based on the configuration settings.
        :return: dict containing DatFrame
        """
        day_dict = {}
        for day in self.config_dict["DAYS"].keys():
            day_df = pd.DataFrame(index=["00:00"])
            for hour, prog in self.config_dict["DAYS"][day].items():
                day_df.loc[hour, prog.keys()] = prog.values()
            day_dict[day] = day_df

        return day_dict

    def get_dataframe(self, freq="T"):
        """
        Generates and returns the scheduled DataFrame based on the configuration
        settings.

        :param str freq: output DataFrame DateTimeIndex frequency
        """

        day_list = []
        for period in self.config_dict["PERIODS"]:
            period_index = pd.date_range(
                start=period[0][0],
                end=period[0][1],
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
        df.sort_index(inplace=True)
        df = df.fillna(method="bfill")
        df = df.resample("T").bfill()
        df = df.shift(-1)
        df = df.fillna(method="ffill")
        return df.resample(freq).mean()


class GreyWaterConsumption:
    """
    A class to calculate the distribution of greywater
    consumption based on various factors.

    Args:
        n_people (int): Number of people in the household.
        seed (bool, optional): Whether to use a seed for random
         number generation. Defaults to False.
        dish_washer (bool, optional): Whether the household has
        a dishwasher. Defaults to True.
        washing_machine (bool, optional): Whether the household has
        a washing machine. Defaults to True.
        v_water_dish (int, optional): Volume of water used
        by the dishwasher (in liters). Defaults to 13.
        v_water_clothes (int, optional): Volume of water used by
        the washing machine (in liters). Defaults to 50.
        cycles_clothes_pers (int, optional): Number of washing machine
        cycles per person per year. Defaults to 89.
        cycles_dish_pers (int, optional): Number of dishwasher cycles
        per person per year. Defaults to 83.
        duration_dish (int, optional): Duration of a dishwasher
         cycle (in hours). Defaults to 4.
        duration_clothes (int, optional): Duration of a washing
         machine cycle (in hours). Defaults to 2.

    Raises:
        ValueError: If both dish_washer and washing_machine are False.

    Methods:
        get_GWdistribution(start, end):
            Calculates the distribution of greywater
            consumption over a given time period.

            Args:
                start (str): Start date of the time period
                (format: 'YYYY-MM-DD').
                end (str): End date of the time period
                (format: 'YYYY-MM-DD').

            Returns:
                pd.DataFrame: DataFrame containing the greywater
                consumption distribution with timestamps as index.
    """

    def __init__(
        self,
        n_people,
        seed=False,
        dish_washer=True,
        washing_machine=True,
        v_water_dish=13,
        v_water_clothes=50,
        cycles_clothes_pers=89,  # per year
        cycles_dish_pers=83,  # per year
        duration_dish=4,
        duration_clothes=2,
    ):
        self.n_people = n_people
        self.v_water_dish = v_water_dish
        self.v_water_clothes = v_water_clothes
        self.cycles_clothes_pers = cycles_clothes_pers
        self.cycles_dish_pers = cycles_dish_pers
        self.duration_dish = duration_dish
        self.duration_clothes = duration_clothes
        self.seed = seed
        self.dish_washer = dish_washer
        self.washing_machine = washing_machine

        if not dish_washer and not washing_machine:
            raise ValueError(
                "At least one of washing machine or dish washer must be True"
            )

    def get_GWdistribution(self, start, end):
        """
        Calculates the distribution of greywater consumption over a given time period.

        Args:
            start (str): Start date of the time period (format: 'YYYY-MM-DD').
            end (str): End date of the time period (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: DataFrame containing the greywater consumption
            distribution with timestamps as index.
        """

        global dish_distribution, washing_distribution
        date_index = pd.date_range(start=start, end=end, freq="H")
        if self.seed is not None:
            rs = RandomState(MT19937(SeedSequence(self.seed)))
        else:
            rs = RandomState()

        # calculation for dishwasher
        if self.dish_washer is True:
            Qwater_dish = self.v_water_clothes / self.duration_clothes
            dish_distribution = [0] * len(date_index)
            tot_cycles_dish_pers = int(self.cycles_dish_pers / 365 * len(date_index))

            for _ in range(self.n_people):
                k = 0  # number of cycles
                index = rs.randint(0, 120)  # time of start of cycles
                while index < (len(date_index) - 4) and k < tot_cycles_dish_pers:
                    dish_distribution[index] = dish_distribution[index] + Qwater_dish
                    dish_distribution[index + 1] = (
                        dish_distribution[index + 1] + Qwater_dish
                    )
                    dish_distribution[index + 1] = (
                        dish_distribution[index + 2] + Qwater_dish
                    )
                    dish_distribution[index + 1] = (
                        dish_distribution[index + 3] + Qwater_dish
                    )
                    space = rs.randint(72, 120)  # day 3 to day 5
                    index = index + space
                    k = k + 1

        # calculation for washing machine
        if self.washing_machine is True:
            Qwater_clothes = self.v_water_dish / self.duration_dish
            washing_distribution = [0] * len(date_index)
            tot_cycles_dish_pers = int(self.cycles_clothes_pers / 365 * len(date_index))

            for _ in range(self.n_people):
                k = 0
                index = rs.randint(0, 120)
                while index < (len(date_index) - 2) and k < tot_cycles_dish_pers:
                    washing_distribution[index] = (
                        washing_distribution[index] + Qwater_clothes
                    )
                    washing_distribution[index + 1] = washing_distribution[index + 1]
                    space = rs.randint(72, 120)
                    index = index + space
                    k = k + 1

        if self.washing_machine and self.dish_washer:
            data = [dish_distribution, washing_distribution]
            data = list(map(list, zip(*data)))
            columns = ["Q_dish", "Q_washer"]
        elif self.washing_machine and self.dish_washer is not True:
            data = washing_distribution
            columns = ["Q_washer"]
        elif self.washing_machine is not True and self.dish_washer is not True:
            data = []
            columns = []
            print(data, columns)
        else:
            data = dish_distribution
            columns = ["Q_dish"]

        df = pd.DataFrame(data=data, index=date_index, columns=columns)

        return df
