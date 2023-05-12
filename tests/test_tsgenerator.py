import pandas as pd
import numpy as np
from corrai.tsgenerator import DHWaterConsumption
from corrai.tsgenerator import Scheduler
import datetime as dt
from pathlib import Path

FILES_PATH = Path(__file__).parent / "resources"


class TestDHWaterConsumption:
    def test_get_coefficient_calc_from_period(self):
        df = pd.DataFrame(
            index=pd.date_range("2023-01-01 00:00:00", freq="H", periods=8760)
        )
        start = df.index[0]
        end = df.index[-1]

        # test COSTIC
        dhw1 = DHWaterConsumption(n_dwellings=50)
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
        dhw2 = DHWaterConsumption(n_dwellings=50, method="RE2020")
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

    def test_get_coefficient_calc_from_period_2(self):
        dhw = DHWaterConsumption(n_dwellings=50)
        start = dt.datetime(2022, 1, 1, 0, 0, 0)
        end = dt.datetime(2024, 10, 20, 1, 0, 0)
        df = dhw.costic_random_shower_distribution(start=start, end=end, seed=42)

        # Check if the sum of random water consumption  distribution is about equal
        # to total estimated consumption
        total_consoECS_COSTIC = dhw.costic_shower_distribution(start=start, end=end)[
            "consoECS_COSTIC"
        ].sum()
        assert np.isclose(
            df["consoECS_COSTIC_random"].sum(), total_consoECS_COSTIC, rtol=0.05
        )

    def test_re2020_shower_distribution(self):
        dhw = DHWaterConsumption(
            n_dwellings=50, s_moy_dwelling=49.6, s_tot_building=2480, method="RE2020"
        )
        start = dt.datetime(2022, 1, 1, 0, 0, 0)
        end = dt.datetime(2024, 10, 20, 1, 0, 0)
        df = dhw.re2020_shower_distribution(start=start, end=end)

        # check Wednesday in April
        assert np.isclose(df.loc["2023-04-05 00:00", "consoECS_RE2020"], 0, rtol=0.05)
        # check Saturday in August
        assert np.isclose(
            df.loc["2023-08-26 20:00", "consoECS_RE2020"], 285.5, rtol=0.05
        )
        # check Sunday in September
        assert np.isclose(
            df.loc["2023-09-10 08:00", "consoECS_RE2020"], 752.7, rtol=0.05
        )

        dhw = DHWaterConsumption(
            n_dwellings=12, s_moy_dwelling=72, s_tot_building=1000, method="RE2020"
        )
        df = dhw.re2020_shower_distribution(start=start, end=end)

        # check Sunday in September
        assert np.isclose(
            df.loc["2023-09-10 08:00", "consoECS_RE2020"], 261.3, rtol=0.05
        )

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
