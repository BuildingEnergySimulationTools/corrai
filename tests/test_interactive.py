import pandas as pd
from corrai.metrics import cv_rmse
from corrai.interactive import interactive_sample_pcp_lines


class TestInteractive:
    def test_interactive_sample_pcp_lines(self):
        sample_res = [
            (
                {"par1": 1, "par2": 3},
                {},
                pd.DataFrame(
                    {
                        "res1": [1, 2, 3, 4],
                        "res2": [5, 6, 7, 8],
                    },
                    index=pd.date_range("2009-01-01", freq="h", periods=4),
                ),
            ),
            (
                {"par1": 2, "par2": 1},
                {},
                pd.DataFrame(
                    {
                        "res1": [2, 4, 6, 8],
                        "res2": [1, 2, 5, 4],
                    },
                    index=pd.date_range("2009-01-01", freq="h", periods=4),
                ),
            ),
        ]

        reference = pd.Series(
            [1, 2, 3, 4], index=pd.date_range("2009-01-01", freq="h", periods=4)
        )

        interactive_sample_pcp_lines(
            sample_results=sample_res,
            target="res1",
            reference=reference,
            color_by="cv_rmse",
            agg_method={"cv_rmse": (cv_rmse, "res1", reference)},
        )
