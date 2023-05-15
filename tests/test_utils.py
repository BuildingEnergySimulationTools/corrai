import pandas as pd
import numpy as np

import pytest

from corrai.utils import _reshape_1d
from corrai.utils import as_1_column_dataframe
from corrai.utils import check_datetime_index
from corrai.utils import float_to_hour
from corrai.utils import hour_to_float


class TestUtils:
    def test_reshape_1d(self):
        x_in = np.array([[1], [2]])
        np.testing.assert_array_equal(_reshape_1d(x_in), np.array([1, 2]))

        x_in = pd.DataFrame({"a": [1, 2]})
        pd.testing.assert_series_equal(_reshape_1d(x_in), pd.Series([1, 2], name="a"))

    def test_as_1_column_dataframe(self):
        ref = pd.DataFrame({0: [1.0, 2.0]})

        # Test list in
        pd.testing.assert_frame_equal(as_1_column_dataframe([1.0, 2.0]), ref)

        # Test DataFrame in
        pd.testing.assert_frame_equal(as_1_column_dataframe(ref), ref)

        # Test Series in
        pd.testing.assert_frame_equal(as_1_column_dataframe(ref.squeeze()), ref)

        # Test 2D array in
        pd.testing.assert_frame_equal(
            as_1_column_dataframe(np.array([[1.0], [2.0]])), ref
        )

        # Test 1D array in
        pd.testing.assert_frame_equal(as_1_column_dataframe(np.array([1.0, 2.0])), ref)

        with pytest.raises(ValueError):
            as_1_column_dataframe(1)

        with pytest.raises(ValueError):
            as_1_column_dataframe(pd.DataFrame({"a": [1], "b": [2]}))

    def test_check_datetime_index(self):
        with pytest.raises(ValueError):
            check_datetime_index(1)

        with pytest.raises(ValueError):
            check_datetime_index(pd.Series([1, 2]))

    def test_float_to_hour(self):
        assert float_to_hour(1.5) == "01:30"
        assert float_to_hour([1.5]) == ["01:30"]
        np.testing.assert_array_equal(
            float_to_hour(np.array([1.5])), np.array(["01:30"])
        )
        np.testing.assert_array_equal(
            float_to_hour(np.array([[1.5]])), np.array([["01:30"]])
        )
        pd.testing.assert_series_equal(
            float_to_hour(pd.Series([1.5])), pd.Series(["01:30"])
        )
        pd.testing.assert_frame_equal(
            float_to_hour(pd.DataFrame({"a": [2.5]})), pd.DataFrame({"a": ["02:30"]})
        )

    def test_hour_to_float(self):
        assert hour_to_float("01:30") == 1.5
        assert hour_to_float(["01:30"]) == [1.5]
        np.testing.assert_array_equal(
            hour_to_float(np.array(["01:30"])), np.array([1.5])
        )
        np.testing.assert_array_equal(
            hour_to_float(np.array([["01:30"]])), np.array([[1.5]])
        )
        pd.testing.assert_series_equal(
            hour_to_float(pd.Series(["01:30"])), pd.Series([1.5])
        )
        pd.testing.assert_frame_equal(
            hour_to_float(pd.DataFrame({"a": ["02:30"]})), pd.DataFrame({"a": [2.5]})
        )
