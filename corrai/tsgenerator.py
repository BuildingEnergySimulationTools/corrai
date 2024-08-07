import datetime as dt

import numpy as np
import pandas as pd
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


class DomesticWaterConsumption:
    """
    Class for calculating domestic hot and cold water consumption
    in a building based on either RE2020 or COSTIC coefficients.

    Warning1: Note that for a small number of dwellings (less than 6),
    the calculation method for random shower water distribution is
    not adapted: since the method uses COSTIC water flow calculations
    for each hour and then randomly distribute showers, it does not
    work properly for small amount of water (nb of shower per hour < 1
    results in 0 shower). Will result in high underestimation for small
    n_dwellings. Likewise, can result in slight under- or over-
    estimation for high number of dwellings.
    Will be addressed in future work.

    Warning2: Water consumption for showers is limited here by a
    water consumption per dwelling in liters per day, then distributed
    using daily COSTIC coefficients. Although COSTIC coefficients are
    calculated for each day of the year, and are representative of
    higher probability of having showers at certain hours, they do
    not impact the total water consumption.
    Another method might be implemented in the future.

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
    n_people_per_dwelling : int, optional (default=2)
        Number of people per dwelling.
    method : str, optional (default="COSTIC")
        Method for calculating hot water consumption.
        Available options are "COSTIC" or "RE2020".

    Attributes:
    -----------
    df_coefficient : pd.DataFrame
        DataFrame containing the coefficients for hot water consumption, calculated
        either using the "COSTIC" or "RE2020" method.
    df_daily_sum : pd.DataFrame
        DataFrame containing the daily sum of coefficients.
    df_re2020 : pd.DataFrame
        DataFrame containing the hot water consumption for showers
        based on the "RE2020" method.
    df_costic : pd.DataFrame
        DataFrame containing the hot water consumption for showers
        based on the "COSTIC" method.
    df_costic_random : pd.DataFrame
        DataFrame containing a random distribution of hot water
        consumption for showers based on the "COSTIC" method.
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
        Calculates the hot water consumption for showers based on the "COSTIC" method
        within the given time period. The output DataFrame is sampled every hour.
    costic_random_shower_distribution(start, end, optional_columns=False, seed=None):
        Calculates a random distribution of hot water consumption for showers based on
        the "COSTIC" method within the given time period.
        The output DataFrame is sampled every minute.
        Set seed to None for a new random distribution
        each time the method is run. Otherwise, set to an integer
        for the same random distribution.
    re2020_shower_distribution(start, end):
        Calculates the hot water consumption for showers based on the "RE2020" method
        within the given time period. The output DataFrame is sampled every hour.
    appliances_water_distribution(start, end, seed=False, dish_washer=True,
        washing_machine=True, v_water_dish=13, v_water_clothes=50,
        cycles_clothes_pers=89, cycles_dish_pers=83,
        duration_dish=4, duration_clothes=2):
        Calculates the distribution of greywater consumption from
        appliances over a given time period.
    day_randomizer(coefficient, nb_used, volume, seed=None):
        Randomly distributes water consumption over a day based on
        coefficients and usage parameters.
    costic_random_cold_water_distribution(start, end, percent_showers=0.4,
        percent_washbasin=0.13, percent_cook=0.07, percent_dishes=0.04,
        percent_cleaning=0.06, seed=False):
        Calculates a random distribution of cold water consumption
        for various usages based on the "COSTIC" method
        within the given time period.

    Note: Detailed descriptions and explanations for each
    method and attribute can be found in the class docstring.
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
        n_people_per_dwelling=2,
        method="COSTIC",
    ):
        self.n_dwellings = n_dwellings
        self.v_per_dwelling = v_per_dwelling
        self.ratio_bath_shower = ratio_bath_shower
        self.t_shower = t_shower
        self.d_shower = d_shower
        self.s_moy_dwelling = s_moy_dwelling
        self.s_tot_building = s_tot_building
        self.n_people_per_dwelling = n_people_per_dwelling
        self.method = method

        self.df_coefficient = None
        self.df_daily_sum = None
        self.df_re2020 = None
        self.df_costic = None
        self.df_costic_random = None
        self.df_all = None

        self.v_used = self.t_shower * self.d_shower
        self.n_people = self.n_dwellings * self.n_people_per_dwelling
        self.v_liters_day = self.n_dwellings * self.v_per_dwelling
        self.v_shower_bath_per_day = self.ratio_bath_shower * self.v_liters_day

        if self.method == "COSTIC":
            self.coefficients = coefficients_COSTIC
        elif self.method == "RE2020":
            self.coefficients = coefficients_RE2020

    def get_coefficient_calc_from_period(self, start, end):
        """
        Calculates the coefficients for hot water consumption
        based on the chosen method within the given time period.

        Args:
            start (str): Start date of the time period (format: 'YYYY-MM-DD').
            end (str): End date of the time period (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: DataFrame containing the calculated coefficients
            for hot water consumption for each time step
            within the specified time period.
        Raises:
            ValueError: If start or end values are not valid timestamps.
        """

        if not pd.Timestamp(start) or not pd.Timestamp(end):
            raise ValueError("Start and end values must be valid timestamps.")

        date_index = pd.date_range(start=start, end=end, freq="h")
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
        """
        Calculates the hot water consumption for showers based
        on the "COSTIC" method within the given time period.

        Args:
            start (str): Start date of the time period (format: 'YYYY-MM-DD').
            end (str): End date of the time period (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: DataFrame containing the calculated
            hot water consumption for showers with timestamps as index.
        """

        # Concatenation of coefficients and their daily sum
        self.get_coefficient_calc_from_period(start, end)
        df_co = pd.concat([self.df_coefficient, self.df_daily_sum], axis=1)

        # Concatenation of coefficients and their daily sum
        self.get_coefficient_calc_from_period(start, end)
        df_co = pd.concat([self.df_coefficient, self.df_daily_sum], axis=1)
        df_co.ffill(inplace=True)
        df_co = df_co.dropna(axis=0)

        # Calculation of the number of showers per hour
        df_co["Q_ECS_COSTIC"] = (
            df_co["coef"] * self.v_shower_bath_per_day / df_co["coef_daily_sum"]
        )
        return df_co[["Q_ECS_COSTIC"]]

    def costic_random_shower_distribution(
        self, start=None, end=None, optional_columns=False, seed=None
    ):
        """
        Calculates a random distribution of hot water
        consumption for showers based on
        the "COSTIC" method within
        the given time period.

        Args:
            start (str, optional): Start date of the time period (format: 'YYYY-MM-DD').
                Defaults to None.
            end (str, optional): End date of the time period (format: 'YYYY-MM-DD').
                Defaults to None.
            optional_columns (bool, optional): Whether to include
                additional columns in the output DataFrame.
                Defaults to False.
            seed (int, optional): Seed for random number generator.
                If provided, the same distribution
                can be reproduced. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the random
            distribution of hot water consumption for showers
            based on the "COSTIC" method with timestamps as index.
        Raises:
            ValueError: If the method is not "COSTIC".

        Note:
            If `optional_columns` is True, the DataFrame includes
            additional columns like shower duration,
            number of showers per hour, and the
            rounded number of showers.

        Warning:
            The `seed` parameter should be used for
            reproducibility only. Providing the same seed will
            result in the same random distribution.
        """
        if not self.method == "COSTIC":
            raise ValueError("Method has to be COSTIC for random shower distribution")

        if seed is not None:
            rs = RandomState(MT19937(SeedSequence(seed)))
        else:
            rs = RandomState()

        periods = pd.date_range(start=start, end=end, freq="min")
        df_costic = self.costic_shower_distribution(start, end)
        df_costic["nb_shower"] = df_costic["Q_ECS_COSTIC"] / self.v_used
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
                freq="min",
                periods=df_costic.shape[0] * 60,
            ),
            columns=["shower_per_minute"],
        )

        df_costic_random["Q_ECS_COSTIC_rd"] = (
            df_costic_random["shower_per_minute"] * self.v_used / self.t_shower
        )
        df_costic_random["Q_ECS_COSTIC_rd"] = df_costic_random[
            "Q_ECS_COSTIC_rd"
        ].astype(float)

        # TODO : fix dimension issue properly
        df_costic_random = df_costic_random.iloc[:-59]

        self.df_costic_random = df_costic_random

        if optional_columns:
            return df_costic_random
        else:
            return df_costic_random[["Q_ECS_COSTIC_rd"]]

    def re2020_shower_distribution(self, start, end):
        """
        Calculates the hot water consumption for showers
        based on the "RE2020" method within the given time period.

        Args:
            start (str): Start date of the time period (format: 'YYYY-MM-DD').
            end (str): End date of the time period (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: DataFrame containing the
            calculated hot water consumption for showers
            with timestamps as index.
        """
        periods = pd.date_range(start=start, end=end, freq="h")
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
        to_return.columns = ["Q_ECS_RE2020"]
        return to_return

    def appliances_water_distribution(
        self,
        start,
        end,
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
        """
        Calculates the distribution of greywater consumption
        from appliances over a given time period.

        Args:
            start (str): Start date of the time period (format: 'YYYY-MM-DD').
            end (str): End date of the time period (format: 'YYYY-MM-DD').
            seed (int or bool, optional): Seed for random number
                generator. If provided, the same distribution
                can be reproduced. Defaults to False.
            dish_washer (bool, optional): Whether to consider
                the dishwasher's greywater consumption.
                Defaults to True.
            washing_machine (bool, optional): Whether to consider
                the washing machine's greywater consumption.
                Defaults to True.
            v_water_dish (float, optional): Greywater consumption
                per cycle of the dishwasher, in liters.
                Defaults to 13.
            v_water_clothes (float, optional): Greywater consumption
                per cycle of the washing machine, in liters.
                Defaults to 50.
            cycles_clothes_pers (int, optional): Number of washing
                machine cycles per person per year.
                Defaults to 89.
            cycles_dish_pers (int, optional): Number of dishwasher
                cycles per person per year.
                Defaults to 83.
            duration_dish (float, optional): Duration of
                a dishwasher cycle in hours.
                Defaults to 4.
            duration_clothes (float, optional): Duration of
                a washing machine cycle in hours.
                Defaults to 2.

        Returns:
            pd.DataFrame: DataFrame containing the greywater
            consumption distribution with timestamps as index.
        """

        self.v_water_dish = v_water_dish
        self.v_water_clothes = v_water_clothes
        self.cycles_clothes_pers = cycles_clothes_pers
        self.cycles_dish_pers = cycles_dish_pers
        self.duration_dish = duration_dish
        self.duration_clothes = duration_clothes
        self.dish_washer = dish_washer
        self.washing_machine = washing_machine
        self.seed = seed

        if not dish_washer and not washing_machine:
            raise ValueError(
                "At least one of washing machine or dish washer must be True"
            )

        date_index = pd.date_range(start=start, end=end, freq="h")
        if self.seed is not None:
            rs = RandomState(MT19937(SeedSequence(self.seed)))
        else:
            rs = RandomState()

        nb_days = (end - start).days + 1

        # calculation for dishwasher
        if self.dish_washer is True:
            Qwater_dish = self.v_water_dish / self.duration_dish
            dish_distribution = [0] * len(date_index)
            tot_cycles_dish_pers = int(self.cycles_dish_pers / 365 * nb_days)

            for _ in range(self.n_people):
                k = 0  # number of cycles
                index = rs.randint(0, 120)  # time of start of cycles
                while (
                    index < (len(date_index) - self.duration_dish)
                    and k < tot_cycles_dish_pers
                ):
                    for i in range(self.duration_dish):
                        dish_distribution[index + i] = (
                            dish_distribution[index + i] + Qwater_dish
                        )
                    space = rs.randint(72, 120)  # day 3 to day 5
                    index = index + space
                    k = k + 1

        # calculation for washing machine
        if self.washing_machine is True:
            Qwater_clothes = self.v_water_clothes / self.duration_clothes
            washing_distribution = [0] * len(date_index)
            tot_cycles_clot_pers = int(self.cycles_clothes_pers / 365 * nb_days)

            for _ in range(self.n_people):
                k = 0
                index = rs.randint(0, 120)
                while (
                    index < (len(date_index) - self.duration_clothes)
                    and k < tot_cycles_clot_pers
                ):
                    for i in range(self.duration_clothes):
                        washing_distribution[index + i] = (
                            washing_distribution[index + i] + Qwater_clothes
                        )
                    space = rs.randint(72, 120)
                    index = index + space
                    k = k + 1

        if self.washing_machine and self.dish_washer:
            data = [dish_distribution, washing_distribution]
            data = list(map(list, zip(*data)))
            columns = ["Q_dish", "Q_washer"]
        elif self.washing_machine and not self.dish_washer:
            data = washing_distribution
            columns = ["Q_washer"]
        else:
            data = dish_distribution
            columns = ["Q_dish"]

        df = pd.DataFrame(data=data, index=date_index, columns=columns)

        return df

    def day_randomizer(self, coefficient, nb_used, volume, seed=None):
        if seed is not None:
            rs = RandomState(MT19937(SeedSequence(seed)))
        else:
            rs = RandomState()

        list_int = [0] * len(coefficient)
        idx_start = None
        idx_end = None

        for i, value in zip(range(len(coefficient)), coefficient["coef"]):
            if value > 0 and idx_start is None:
                idx_start = i
            if idx_start is not None and value == 0:
                idx_end = i - 1
                break

        if idx_end is None:
            idx_end = len(coefficient) - 1

        for _ in range(self.n_dwellings * nb_used):
            k = rs.randint(low=idx_start, high=idx_end + 1)
            list_int[k] += volume
        return list_int

    def costic_random_cold_water_distribution(
        self,
        start,
        end,
        percent_showers=0.4,
        percent_washbasin=0.13,
        percent_cook=0.07,
        percent_dishes=0.04,
        percent_cleaning=0.06,
        seed=False,
    ):
        """
        Calculates a random distribution of cold water consumption for various usages
        based on the "COSTIC" method within the given time period.

        Args:
            start (str): Start date of the time period (format: 'YYYY-MM-DD').
            end (str): End date of the time period (format: 'YYYY-MM-DD').
            percent_showers (float, optional): Percentage of
                hot water consumption dedicated to showers.
                Defaults to 0.4.
            percent_washbasin (float, optional): Percentage of hot water
                consumption dedicated to washbasin usage.
                Defaults to 0.13.
            percent_cook (float, optional): Percentage of hot water
                consumption dedicated to cooking-related usage.
                Defaults to 0.07.
            percent_dishes (float, optional): Percentage of hot
                water consumption dedicated to dishwashing.
                Defaults to 0.04.
            percent_cleaning (float, optional): Percentage of hot
                water consumption dedicated to cleaning purposes.
                Defaults to 0.06.
            seed (int or bool, optional): Seed for random number
                generator. If provided, the same distribution
                can be reproduced. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the random
                distribution of cold water consumption
            for different usages with timestamps as index.
        """

        if not self.method == "COSTIC":
            raise ValueError(
                "Method has to be COSTIC for random cold water distribution"
            )

        self.percent_showers = percent_showers
        self.percent_washbasin = percent_washbasin
        self.percent_cook = percent_cook
        self.percent_dishes = percent_dishes
        self.percent_cleaning = percent_cleaning
        self.seed = seed

        # Volumes = liters per usage
        self.v_washbasin_used = 10
        self.v_sinkcook_used = 10.5
        self.v_sinkdishes_used = 30
        self.v_sinkcleaning_used = 8.8

        # Total consumed Liters/day per dwelling
        self.v_water_tot = self.v_per_dwelling / self.percent_showers

        #  Consumed Liters/day per dwelling for each usage
        self.v_washbasin = self.v_water_tot * self.percent_washbasin
        self.v_sink_cook = self.v_water_tot * self.percent_cook
        self.v_sink_dishes = self.v_water_tot * self.percent_dishes
        self.v_sink_cleaning = self.v_water_tot * self.percent_cleaning

        nb_washbasin = round(self.v_washbasin / self.v_washbasin_used)
        nb_sinkcook = round(self.v_sink_cook / self.v_sinkcook_used)
        nb_sinkdishes = round(self.v_sink_dishes / self.v_sinkdishes_used)
        nb_sinkcleaning = round(self.v_sink_cleaning / self.v_sinkcleaning_used)

        coef = self.get_coefficient_calc_from_period(start, end)
        coefficient = coef.mask(coef["coef"] < 0.5)
        coefficient = coefficient.resample("min").ffill().fillna(float("0"))

        list_washbasin = []
        list_sinkcook = []
        list_sinkdishes = []
        list_sinkwash = []

        for _ in range((end - start).days):
            list_washbasin += self.day_randomizer(
                coefficient=coefficient[
                    coefficient.index.date == coefficient.index.date[0]
                ],
                nb_used=nb_washbasin,
                volume=self.v_washbasin_used,
                seed=seed,
            )
            list_sinkcook += self.day_randomizer(
                coefficient=coefficient[
                    coefficient.index.date == coefficient.index.date[0]
                ],
                nb_used=nb_sinkcook,
                volume=self.v_sinkcook_used,
                seed=seed,
            )

            list_sinkdishes += self.day_randomizer(
                coefficient=coefficient[
                    coefficient.index.date == coefficient.index.date[0]
                ],
                nb_used=nb_sinkdishes,
                volume=self.v_sinkdishes_used,
                seed=seed,
            )

            list_sinkwash += self.day_randomizer(
                coefficient=coefficient[
                    coefficient.index.date == coefficient.index.date[0]
                ],
                nb_used=nb_sinkcleaning,
                volume=self.v_sinkcleaning_used,
            )

        list_washbasin.append(0)
        list_sinkcook.append(0)
        list_sinkdishes.append(0)
        list_sinkwash.append(0)

        df_co = coefficient.copy()
        df_co["Q_washbasin_COSTIC_rd"] = list_washbasin
        df_co["Q_sink_cook_COSTIC_rd"] = list_sinkcook
        df_co["Q_sink_dishes_COSTIC_rd"] = list_sinkdishes
        df_co["Q_sink_cleaning_COSTIC_rd"] = list_sinkwash

        # df_co.drop(df_co.index[-1], inplace=True)

        return df_co[
            [
                "Q_washbasin_COSTIC_rd",
                "Q_sink_cook_COSTIC_rd",
                "Q_sink_dishes_COSTIC_rd",
                "Q_sink_cleaning_COSTIC_rd",
            ]
        ]


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
                    (("01-01", "03-31"), "winter_week"),
                    (("04-01", "09-30"), "summer_week"),
                    (("10-01", "12-31"), "winter_week"),
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
            day_df = pd.DataFrame.from_dict(
                self.config_dict["DAYS"][day], orient="index"
            )
            if "00:00" not in day_df.index:
                day_df.loc["00:00", :] = [np.nan] * len(day_df.columns)
            day_dict[day] = day_df

        return day_dict

    def get_full_year_time_series(self, year=None, freq="min"):
        """
        Generates and returns the scheduled DataFrame based on the configuration
        settings.

        :param year: Year to generate the time series for. Default is current year
        :param str freq: output DataFrame DateTimeIndex frequency
        """

        if year is None:
            year = dt.datetime.now().year

        day_list = []
        for period in self.config_dict["PERIODS"]:
            period_index = pd.date_range(
                start=f"{str(year)}-{period[0][0]}",
                end=f"{str(year)}-{period[0][1]}",
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
                    current_day.index = new_index.tz_localize(self.config_dict["TZ"])
                    day_list.append(current_day)

        df = pd.concat(day_list)
        df.sort_index(inplace=True)
        df = df.bfill()
        df = df.resample("min").bfill()
        df = df.shift(-1)
        df = df.ffill()
        return df.resample(freq).mean()


def calculate_power(
    df,
    deltaT,
    Cp=4180,
    ro=1,
):
    """
    Calculate power consumption based on given DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the data for power calculation.
        The flow rates of water should be in L/h.
        deltaT (float): Temperature difference (in Kelvin).
        Cp (float): Specific heat capacity of the fluid (4180 J/(kg·Kelvin)
            for water by default).
        ro (float): Density of medium (1 kg/L for water by default )

    Returns:
        DataFrame: DataFrame containing the calculated power consumption
        for each column in the input DataFrame, expressed in kW.
    """
    df_powers = df * Cp * ro * deltaT / 3.6e6
    df_powers.columns = ["P_" + col for col in df.columns + "(kW)"]

    return df_powers


def resample_flow_rate(df, new_freq):
    """
    Resample the flow rate hour columns of DataFrame to a new frequency.

    Parameters:
        df (DataFrame): DataFrame containing the flow
        rate data to be resampled.
        new_freq (str): New frequency desired for resampling
        (e.g., '30T' for every 30 minutes).

    Returns:
        DataFrame: DataFrame with flow rate columns resampled to the new frequency.

    """
    original_freq = df.index.freqstr
    if original_freq is None:
        raise ValueError(
            "DataFrame index does not have a frequency. "
            "Please set the frequency before resampling."
        )

    resampled_df = df.resample(new_freq).first()

    if original_freq is not None:
        original_minutes = pd.Timedelta(df.index[1] - df.index[0]).total_seconds() / 60
        new_minutes = pd.Timedelta(new_freq).total_seconds() / 60
        ratio = original_minutes / new_minutes

        resampled_df = resampled_df / ratio
    resampled_df = resampled_df.interpolate(method="linear")

    return resampled_df
