import pandas as pd
import numpy as np
from corrai.tsgenerator import DomesticWaterConsumption
from corrai.tsgenerator import Scheduler
import datetime as dt
from pathlib import Path
import pytest

FILES_PATH = Path(__file__).parent / "resources"


class TestDomesticWaterConsumption:
    def test_get_coefficient_calc_from_period(self):
        df = pd.DataFrame(
            index=pd.date_range("2023-01-01 00:00:00", freq="H", periods=8760)
        )
        start = df.index[0]
        end = df.index[-1]

        # test COSTIC
        dhw1 = DomesticWaterConsumption(n_dwellings=50)
        dhw1.get_coefficient_calc_from_period(start=start, end=end)
        assert round(dhw1.df_coefficient.loc["2023-04-05 00:00", "coef"], 4) == round(
            1.06 * 0.264 * 1.00, 4
        )
        assert round(dhw1.df_coefficient.loc["2023-08-26 20:00", "coef"], 4) == round(
            1.392 * 0.72 * 1.02, 4
        )
        assert round(dhw1.df_coefficient.loc["2023-09-10 11:00", "coef"], 4) == round(
            1.752 * 0.92 * 1.13, 4
        )

        # #test RE2020
        dhw2 = DomesticWaterConsumption(n_dwellings=50, method="RE2020")
        dhw2.get_coefficient_calc_from_period(start=start, end=end)
        assert round(dhw2.df_coefficient.loc["2023-04-05 00:00", "coef"], 4) == round(
            0 * 0.95, 4
        )
        assert round(dhw2.df_coefficient.loc["2023-08-25 20:00", "coef"], 4) == round(
            0.022 * 0.95, 4
        )
        assert round(dhw2.df_coefficient.loc["2023-09-10 18:00", "coef"], 4) == round(
            0.011 * 0.95, 4
        )

    def test_costic_random_distribution(self):
        dhw = DomesticWaterConsumption(n_dwellings=50)
        start = dt.datetime(2022, 1, 1, 0, 0, 0)
        end = dt.datetime(2024, 10, 20, 1, 0, 0)
        df = dhw.costic_random_shower_distribution(start=start, end=end, seed=42)
        df2 = dhw.costic_random_shower_distribution(start=start, end=end, seed=None)

        # Check if the sum of random water consumption  distribution is about equal
        # to total estimated consumption
        total_consoECS_COSTIC = dhw.costic_shower_distribution(start=start, end=end)[
            "Q_ECS_COSTIC"
        ].sum()
        assert np.isclose(df["Q_ECS_COSTIC_rd"].sum(), total_consoECS_COSTIC, rtol=0.05)
        assert np.isclose(
            df2["Q_ECS_COSTIC_rd"].sum(), total_consoECS_COSTIC, rtol=0.05
        )

    def test_re2020_shower_distribution(self):
        dhw = DomesticWaterConsumption(
            n_dwellings=50, s_moy_dwelling=49.6, s_tot_building=2480, method="RE2020"
        )
        start = dt.datetime(2022, 1, 1, 0, 0, 0)
        end = dt.datetime(2024, 10, 20, 1, 0, 0)
        df = dhw.re2020_shower_distribution(start=start, end=end)

        # check Wednesday in April
        assert np.isclose(df.loc["2023-04-05 00:00", "Q_ECS_RE2020"], 0, rtol=0.05)
        # check Saturday in August
        assert np.isclose(df.loc["2023-08-26 20:00", "Q_ECS_RE2020"], 285.5, rtol=0.05)
        # check Sunday in September
        assert np.isclose(df.loc["2023-09-10 08:00", "Q_ECS_RE2020"], 752.7, rtol=0.05)

        dhw = DomesticWaterConsumption(
            n_dwellings=12, s_moy_dwelling=72, s_tot_building=1000, method="RE2020"
        )
        df = dhw.re2020_shower_distribution(start=start, end=end)

        # check Sunday in September
        assert np.isclose(df.loc["2023-09-10 08:00", "Q_ECS_RE2020"], 261.3, rtol=0.05)

    def test_appliances_water_distribution(self):
        dhw = DomesticWaterConsumption(n_dwellings=6)

        df = pd.DataFrame(
            index=pd.date_range("2020-01-01 00:00:00", freq="H", periods=8760)
        )
        start = df.index[0]
        end = df.index[-1]

        gw = dhw.appliances_water_distribution(start=start, end=end, seed=42)

        assert np.isclose(gw["Q_dish"].sum(), 54850.0, rtol=0.05)

        assert np.isclose(gw["Q_washer"].sum(), 3565.25, rtol=0.05)

    def test_day_randomizer(self):
        gw = DomesticWaterConsumption(n_dwellings=100, method="COSTIC")

        day1 = 1
        day2 = 20
        # TODO: nb days related to dimension issue
        nb_day = day2 - day1 + 2

        start = dt.datetime(2022, 1, day1, 0, 0, 0)
        end = dt.datetime(2022, 1, day2, 0, 0, 0)

        # Calculate the expected sum
        test = gw.costic_random_cold_water_distribution(start, end)
        expected_sum = round(gw.v_washbasin) * gw.n_dwellings * nb_day

        # Calculate the real sum
        real_sum = sum(test["Q_washbasin_COSTIC_rd"])
        # Verify that the sum of the result_list matches the expected_sum
        assert np.isclose(real_sum, expected_sum, rtol=0.05)

    def test_warning_errors(self):
        df = pd.DataFrame(
            index=pd.date_range("2020-01-01 00:00:00", freq="H", periods=8760)
        )
        start = df.index[0]
        end = df.index[-1]

        dhw0 = DomesticWaterConsumption(n_dwellings=6, method="RE2020")

        with pytest.raises(ValueError):
            dhw0.costic_random_cold_water_distribution(start=start, end=end)

        with pytest.raises(ValueError):
            dhw0.costic_random_cold_water_distribution(start="wrongdate", end=end)

        dhw1 = DomesticWaterConsumption(n_dwellings=6)

        with pytest.raises(ValueError):
            dhw1.appliances_water_distribution(
                dish_washer=False, washing_machine=False, start=start, end=end
            )

        with pytest.raises(ValueError):
            dhw0.costic_random_shower_distribution(start=start, end=end)

    def test_scheduler(self):
        schedule_dict = {
            "DAYS": {
                "working_day": {
                    "09:15": {"heating": 17, "extraction_flow_rate": 0},
                    "18:00": {"heating": 21, "extraction_flow_rate": 3000},
                    "19:00": {"heating": 22},
                    "23:00": {"heating": 17, "extraction_flow_rate": 0},
                },
                "Off": {
                    "23:00": {"heating": 17, "extraction_flow_rate": 0},
                },
            },
            "WEEKS": {
                "winter_week": {
                    "Monday": "working_day",
                    "Tuesday": "working_day",
                    "Wednesday": "working_day",
                    "Thursday": "working_day",
                    "Friday": "working_day",
                    "Saturday": "Off",
                    "Sunday": "Off",
                },
                "summer_week": {
                    "Monday": "Off",
                    "Tuesday": "Off",
                    "Wednesday": "Off",
                    "Thursday": "Off",
                    "Friday": "Off",
                    "Saturday": "Off",
                    "Sunday": "Off",
                },
            },
            "PERIODS": [
                (("2009-01-01", "2009-03-31"), "winter_week"),
                (("2009-04-01", "2009-09-30"), "summer_week"),
                (("2009-10-01", "2009-12-31"), "winter_week"),
            ],
            "TZ": "Europe/Paris",
        }

        ref = pd.read_csv(
            filepath_or_buffer=FILES_PATH / "scheduler_test_ref.csv",
            parse_dates=True,
            index_col=0,
        )

        ref.index = pd.date_range(ref.index[0], periods=ref.shape[0], freq="H")

        sched = Scheduler(config_dict=schedule_dict)

        df = sched.get_dataframe(freq="H")

        pd.testing.assert_frame_equal(df, ref)
