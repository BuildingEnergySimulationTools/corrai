import pandas as pd
import numpy as np
import datetime as dt
from functools import partial
from abc import ABC, abstractmethod
from collections.abc import Callable

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from scipy.ndimage import gaussian_filter1d

from corrai.base.math import time_gradient
from corrai.base.utils import (
    get_data_blocks,
    get_outer_timestamps,
    check_and_return_dt_index_df,
)
from corrai.learning.error_detection import STLEDetector, SkSTLForecast

MODEL_MAP = {"STL": SkSTLForecast}


class PdTransformerBC(TransformerMixin, BaseEstimator, ABC):
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, attributes=["features_"])
        return self.features_

    @abstractmethod
    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        """Operations happening during fitting process"""
        pass

    @abstractmethod
    def transform(self, X):
        """Operations happening during transforming process"""
        pass


class PdIdentity(PdTransformerBC):
    """
    A custom transformer that returns the input data without any modifications.

    This transformer is useful when you want to include an identity transformation step
    in a scikit-learn pipeline, where the input data should be returned unchanged.

    Parameters:
    -----------
    None

    Methods:
    --------
    fit(X, y=None):
        This method does nothing and simply returns the transformer instance.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,), optional (default=None)
            The target values.

        Returns:
        --------
        self : object
            The transformer instance itself.

    transform(X):
        This method returns the input data without any modifications.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        transformed_X : array-like, shape (n_samples, n_features)
            The input data without any modifications.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index_ = X.index
        return self

    def transform(self, X):
        return check_and_return_dt_index_df(X)


class PdReplaceDuplicated(PdTransformerBC):
    """This transformer replaces duplicated values in each column by
    specified new value.

    Parameters
    ----------
    keep : str, default 'first'
        Specify which of the duplicated (if any) value to keep.
        Allowed arguments : ‘first’, ‘last’, False.

    Attributes
    ----------
    value : str, default np.nan
        value used to replace not kept duplicated.

    Methods
    -------
    fit(X, y=None)
        Returns self.

    transform(X)
        Drops the duplicated values in the Pandas DataFrame `X`
        Returns the DataFrame with the duplicated filled with 'value'
    """

    def __init__(self, keep="first", value=np.nan):
        super().__init__()
        self.keep = keep
        self.value = value

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index_ = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        X = check_and_return_dt_index_df(X)
        for col in X.columns:
            X.loc[X[col].duplicated(keep=self.keep), col] = self.value
        return X


class PdDropna(PdTransformerBC):
    """A class to drop NaN values in a Pandas DataFrame.

    Parameters
    ----------
    how : str, default 'all'
        How to drop missing values in the data. 'all' drops the row/column if
        all the values are missing, 'any' drops the row/column if any value is
        missing, and a number 'n' drops the row/column if there are at least
        'n' missing values.

    Attributes
    ----------
    how : str
        How to drop missing values in the data.

    Methods
    -------
    fit(X, y=None)
        Returns self.

    transform(X)
        Drops the NaN values in the Pandas DataFrame `X` based on the `how`
        attribute.
        Returns the DataFrame with the NaN values dropped.
    """

    def __init__(self, how="all"):
        super().__init__()
        self.how = how

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        return check_and_return_dt_index_df(X).dropna(how=self.how)


class PdRenameColumns(PdTransformerBC):
    """
    Scikit-learn transformer that renames columns of a Pandas DataFrame.

    Parameters
    ----------
    new_names: list or dict
        A list or a dictionary of new names for columns of a DataFrame.
        If it is a list, it must have the same length as the number of columns
        in the DataFrame. If it is a dictionary, keys must be the old names of
        columns and values must be the new names.

    Attributes
    ----------
    new_names: list or dict
        A list or a dictionary of new names for columns of a DataFrame.

    Methods
    -------
    fit(self, x, y=None)
       No learning is performed, the method simply returns self.

    transform(self, x)
        Renames columns of a DataFrame.

    inverse_transform(self, x)
        Renames columns of a DataFrame.
    """

    def __init__(self, new_names: list[str] | dict[str, str]):
        super().__init__()
        self.new_names = new_names

    def fit(self, X, y=None):
        self.features_ = X.columns
        self.index_ = X.index

        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["features_", "index_"])
        if isinstance(self.new_names, list):
            if len(self.new_names) != len(X.columns):
                raise ValueError(
                    "Length of new_names list must match the number "
                    "of columns in the DataFrame."
                )
            X.columns = self.new_names
        elif isinstance(self.new_names, dict):
            X.rename(columns=self.new_names, inplace=True)
        return X

    def inverse_transform(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["features_", "index_"])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X.columns = self.features_
        return self.transform(X)


class PdSkTransformer(PdTransformerBC):
    """A transformer class to apply scikit transformers on a pandas DataFrame

    This class takes in a scikit-learn transformers as input and applies the
    transformer to a pandas DataFrame. The resulting data will be a pandas
    DataFrame with the same index and columns as the input DataFrame.

    Parameters
    ----------
    transformer : object
        A scikit-learn transformer to apply on the data.

    Attributes
    ----------
    transformer : object
        A scikit-learn transformer that is fitted on the data.

    Methods
    -------
    fit(x, y=None)
        Fit the scaler to the input data `x`

    transform(x)
        Apply the transformer to the input data `x` and return the result
        as a pandas DataFrame.

    inverse_transform(x)
        Apply the inverse transformer to the input data `x` and return the
        result as a pandas DataFrame.

    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.transformer.fit(X)
        self.features_ = X.columns
        self.index_ = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["features_", "index_"])
        return pd.DataFrame(
            data=self.transformer.transform(X), index=X.index, columns=X.columns
        )

    def inverse_transform(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["features_", "index_"])
        X = check_and_return_dt_index_df(X)
        return pd.DataFrame(
            data=self.transformer.inverse_transform(X), index=X.index, columns=X.columns
        )


class PdDropThreshold(PdTransformerBC):
    """Class replacing values in a pandas DataFrame by NaN based on
    threshold values.

    This class implements the scikit-learn transformer API and can be used in
    a scikit-learn pipeline.

    Parameters
    ----------
    upper : float, optional (default=None)
        The upper threshold for values in the DataFrame. Values greater than
        The upper threshold for values in the DataFrame. Values greater than
        this threshold will be replaced.
    lower : float, optional (default=None)
        The lower threshold for values in the DataFrame. Values less than
        this threshold will be replaced.

    Attributes
    ----------
    lower : float
        The lower threshold for values in the DataFrame.
    upper : float
        The upper threshold for values in the DataFrame.

    Methods
    -------
    fit(self, X, y=None)
        No learning is performed, the method simply returns self.

    transform(self, X)
        Transforms the input DataFrame by replacing values based on the
        specified upper and lower thresholds.
    """

    def __init__(self, upper=None, lower=None):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index_ = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        X = check_and_return_dt_index_df(X)
        if self.lower is not None:
            lower_mask = X < self.lower
        else:
            lower_mask = pd.DataFrame(
                np.full(X.shape, False), index=X.index, columns=X.columns
            )

        if self.upper is not None:
            upper_mask = X > self.upper
        else:
            upper_mask = pd.DataFrame(
                np.full(X.shape, False), index=X.index, columns=X.columns
            )

        X[np.logical_or(lower_mask, upper_mask)] = np.nan

        return X


class PdDropTimeGradient(PdTransformerBC):
    """
    A transformer that removes values in a DataFrame based on the time gradient.

    The time gradient is calculated as the difference of consecutive values in
    the time series divided by the time delta between each value.
    If the gradient is below the `lower_rate` or above the `upper_rate`,
    then the value is set to NaN.

    Parameters
    ----------
    dropna : bool, default=True
        Whether to remove NaN values from the DataFrame before processing.
    upper_rate : float, optional
        The upper rate threshold. If the gradient is greater than or equal to
        this value, the value will be set to NaN.
    lower_rate : float, optional
        The lower rate threshold. If the gradient is less than or equal to
         this value, the value will be set to NaN.

    Attributes
    ----------
    None

    Methods
    -------
    fit(X, y=None)
        No learning is performed, the method simply returns self.
    transform(X)
        Removes values in the DataFrame based on the time gradient.

    Returns
    -------
    DataFrame
        The transformed DataFrame.
    """

    def __init__(self, dropna=True, upper_rate=None, lower_rate=None):
        super().__init__()
        self.dropna = dropna
        self.upper_rate = upper_rate
        self.lower_rate = lower_rate

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index_ = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        X = check_and_return_dt_index_df(X)
        X_transformed = []
        for column in X.columns:
            X_column = X[column]
            if self.dropna:
                original_index = X_column.index.copy()
                X_column = X_column.dropna()

            time_delta = X_column.index.to_series().diff().dt.total_seconds()
            abs_der = abs(X_column.diff().divide(time_delta, axis=0))
            abs_der_two = abs(X_column.diff(periods=2).divide(time_delta, axis=0))
            if self.upper_rate is not None:
                mask_der = abs_der >= self.upper_rate
                mask_der_two = abs_der_two >= self.upper_rate
            else:
                mask_der = pd.Series(
                    np.full(X_column.shape, False),
                    index=X_column.index,
                    name=X_column.name,
                )
                mask_der_two = mask_der

            if self.lower_rate is not None:
                mask_constant = abs_der <= self.lower_rate
            else:
                mask_constant = pd.Series(
                    np.full(X_column.shape, False),
                    index=X_column.index,
                    name=X_column.name,
                )

            mask_to_remove = np.logical_and(mask_der, mask_der_two)
            mask_to_remove = np.logical_or(mask_to_remove, mask_constant)

            X_column[mask_to_remove] = np.nan
            if self.dropna:
                X_column = X_column.reindex(original_index)
            X_transformed.append(X_column)
        return pd.concat(X_transformed, axis=1)


class PdApplyExpression(PdTransformerBC):
    """A transformer class to apply a mathematical expression on a Pandas
    DataFrame.

    This class implements a transformer that can be used to apply a
     mathematical expression to a Pandas DataFrame.
    The expression can be any valid Python expression that
    can be evaluated using the `eval` function.

    Parameters
    ----------
    expression : str
        A string representing a valid Python expression.
        The expression can use any variables defined in the local scope,
        including the `X` variable that is passed to the `transform` method
         as the input data.

    Attributes
    ----------
    expression : str
        The mathematical expression that will be applied to the input data.

    """

    def __init__(self, expression):
        super().__init__()
        self.expression = expression

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        X = check_and_return_dt_index_df(X)
        return eval(self.expression)


class PdTimeGradient(PdTransformerBC):
    """
    A class to calculate the time gradient of a pandas DataFrame,
     which is the derivative of the data with respect to time.

    Parameters
    ----------
    dropna : bool, optional (default=True)
        Whether to drop NaN values before calculating the time gradient.

    Attributes
    ----------
    dropna : bool
        The dropna attribute of the class.

    Methods
    -------
    fit(X, y=None)
        Fits the transformer to the data. Does not modify the input data.

    transform(X)
        Transforms the input data by calculating the time gradient of
         the data.

    """

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        X = check_and_return_dt_index_df(X)
        original_index = X.index.copy()
        derivative = time_gradient(X)
        return derivative.reindex(original_index)


class PdFfill(PdTransformerBC):
    """
    A class to front-fill missing values in a Pandas DataFrame.
    the limit argument allows the function to stop frontfilling at a certain
    number of missing value

    Parameters:
        limit: int, default None If limit is specified, this is the maximum number
        of consecutive NaN values to forward/backward fill.
        In other words, if there is a gap with more than this number of consecutive
        NaNs, it will only be partially filled.
        If limit is not specified, this is the maximum number of entries along
        the entire axis where NaNs will be filled. Must be greater than 0 if not None.

    Methods:
        fit(self, X, y=None):
            Does nothing. Returns the object itself.
        transform(self, X):
            Fill missing values in the input DataFrame.
    """

    def __init__(self, limit: int = None):
        super().__init__()
        self.limit = limit

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        X = check_and_return_dt_index_df(X)
        return X.ffill(limit=self.limit)


class PdBfill(PdTransformerBC):
    """
    A class to back-fill missing values in a Pandas DataFrame.
    the limit argument allows the function to stop backfilling at a certain
    number of missing value

    Parameters:
        limit: int, default None If limit is specified, this is the maximum number
        of consecutive NaN values to forward/backward fill.
        In other words, if there is a gap with more than this number of consecutive
        NaNs, it will only be partially filled.
        If limit is not specified, this is the maximum number of entries along
        the entire axis where NaNs will be filled. Must be greater than 0 if not None.

    Methods:
        fit(self, X, y=None):
            Does nothing. Returns the object itself.
        transform(self, X):
            Fill missing values in the input DataFrame.
    """

    def __init__(self, limit: int = None):
        super().__init__()
        self.limit = limit

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        X = check_and_return_dt_index_df(X)
        return X.bfill(limit=self.limit)


class PdFillNa(PdTransformerBC):
    """
    A class that extends scikit-learn's TransformerMixin and BaseEstimator
    to fill missing values in a Pandas DataFrame.

    Parameters:
        value: scalar, dict, Series, or DataFrame
            Value(s) used to replace missing values.

    Methods:
        fit(self, X, y=None):
            Does nothing. Returns the object itself.
        transform(self, X):
            Fill missing values in the input DataFrame.
    """

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index_ = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        X = check_and_return_dt_index_df(X)
        return X.fillna(self.value)


class PdResampler(PdTransformerBC):
    """A transformer class that resamples a Pandas DataFrame

    This class performs resampling on a Pandas DataFrame using the specified
    `rule` and `method`. The resampling operation aggregates the data by time
    intervals of the given `rule` and applies the specified `method` to the
    data within each interval.

    This transformer extends the scikit-learn `TransformerMixin` and `BaseEstimator`
    classes.

    Parameters:
    -----------
    rule : str
        The frequency of resampling. The `rule` must be a string that can be
        passed to the `resample()` method of a Pandas DataFrame.
        It can also be a datetime.timedelta object
    method : str, callable, or list of str/callable, default None
        The aggregation method to be used in resampling. The `method`
        must be a string or callable that can be passed to the `agg()` method
        of a resampled Pandas DataFrame.

    Attributes:
    ----------
    columns : array-like
        The names of the columns in the input data.
    resampled_index : DatetimeIndex
        The index of the resampled Pandas DataFrame.

    Methods:
    -------
    get_feature_names_out(input_features=None)
        Returns the names of the output features.
    fit(X, y=None)
        Fit the transformer to the input data.
    transform(X)
        Perform the resampling operation on the input data and return the
        resampled DataFrame.

    Returns:
    -------
    X_resampled : Pandas DataFrame
        The resampled Pandas DataFrame.
    """

    def __init__(self, rule: str | pd.Timedelta | dt.timedelta, method=None):
        super().__init__()
        self.rule = rule
        self.method = method

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.resampled_index_ = X.resample(self.rule).asfreq()
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        X = check_and_return_dt_index_df(X)
        check_is_fitted(self, attributes=["resampled_index_", "features_"])
        X = X.apply(pd.to_numeric)
        X_resampled = X.resample(self.rule).agg(self.method)
        self.resampled_index_ = X_resampled.index
        return X_resampled


class PdInterpolate(PdTransformerBC):
    """A class that implements interpolation of missing values in
     a Pandas DataFrame.

    This class is a transformer that performs interpolation of missing
    values in a Pandas DataFrame, using the specified `method`.

    Parameters:
    -----------
    method : str or None, default None
        The interpolation method to use. If None, the default interpolation
         method of the Pandas DataFrame `interpolate()` method will be used.

    Attributes:
    -----------
    columns : Index or None
        The columns of the input DataFrame. Will be set during fitting.
    index : Index or None
        The index of the input DataFrame. Will be set during fitting.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the input DataFrame X. This method will set
         the `columns` and `index` attributes of the transformer,
          and return the transformer instance.
    transform(X):
        Transform the input DataFrame X by performing interpolation of
         missing values using the
        specified `method`. Returns the transformed DataFrame.

    Returns:
    -------
    A transformed Pandas DataFrame with interpolated missing values.
    """

    def __init__(self, method=None):
        super().__init__()
        self.method = method

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index_ = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        X = check_and_return_dt_index_df(X)
        return X.interpolate(method=self.method)


class PdColumnResampler(PdTransformerBC):
    """Resample time series data in a pandas DataFrame based on different
    resampling methods for different columns.

    WARNING Use PdResampler if you want to resample all the columns with
    the same aggregation method

    Parameters
    ----------
    rule : str
        The offset string or object representing the target resampling
        frequency.
    columns_method : list of tuples
        List of tuples containing a list of column names and an associated
        resampling method. The method should be a callable that can be passed
        to the `agg()` method of a pandas DataFrame.
    remainder : str or callable, default='drop'
        If 'drop', drop the non-aggregated columns from the resampled output.
        If callable, must be a function that takes a DataFrame as input and
        returns a DataFrame. Any non-aggregated columns not included in the
        output of this function will be dropped from the resampled output.

    Attributes
    ----------
    resampled_index : pandas.DatetimeIndex or None
        Index of the resampled data after fitting.
    columns : list of str or None
        List of columns used to fit the transformer. If `remainder` is set to
        'drop', this will only contain the columns used for aggregation.
        Otherwise, it will contain all columns in the original DataFrame.


    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input DataFrame.
    transform(X)
        Resample the input DataFrame based on the specified resampling methods.
    get_feature_names_out()
        Return the names of the new features created in `transform()`.

    """

    def __init__(self, rule, columns_method, remainder: str | Callable = "drop"):
        super().__init__()
        self.rule = rule
        self.columns_method = columns_method
        self.resampled_index = None
        self.remainder = remainder
        self._check_columns_method()

    def _check_columns_method(self):
        if not isinstance(self.columns_method, list):
            raise ValueError(
                "Columns_method must be a list of Tuple"
                "first index shall be a list of columns names,"
                "second index shall be an aggregation method callable or pandas"
                "Groupby method name"
            )
        for elmt in self.columns_method:
            if not isinstance(elmt[0], list):
                raise ValueError("Tuple first element must be a list" "of columns")

    def _check_columns(self, X):
        for col_list, _ in self.columns_method:
            for col in col_list:
                if col not in X.columns:
                    raise ValueError("Columns in columns_method not found in" "X")

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self._check_columns(X)
        if self.remainder == "drop":
            self.features_ = []
            for col_list, _ in self.columns_method:
                self.features_ += col_list
        else:
            self.features_ = X.columns

        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["features_"])
        X = check_and_return_dt_index_df(X)
        if self.columns_method:
            transformed_X = pd.concat(
                [
                    X[tup[0]].resample(self.rule).agg(tup[1])
                    for tup in self.columns_method
                ],
                axis=1,
            )
        else:
            transformed_X = pd.DataFrame()

        if self.remainder != "drop":
            remaining_col = [
                col for col in X.columns if col not in transformed_X.columns
            ]
            transformed_X = pd.concat(
                [
                    transformed_X,
                    X[remaining_col].resample(self.rule).agg(self.remainder),
                ],
                axis=1,
            )

        return transformed_X[self.features_]


class PdAddTimeLag(PdTransformerBC):
    """
     PdAddTimeLag - A transformer that adds lagged features to a pandas
     DataFrame.

    This transformer creates new features based on the provided features
    lagged by the given time lag.

    Parameters:
    -----------
    time_lag : datetime.timedelta
        The time lag used to shift the provided features. A positive time lag
        indicates that the new features will contain information from the past,
         while a negative time lag indicates that the new features will
        contain information from the future.

    features_to_lag : list of str or str or None, optional (default=None)
        The list of feature names to lag. If None, all features in the input
         DataFrame will be lagged.

    feature_marker : str or None, optional (default=None)
        The string used to prefix the names of the new lagged features.
        If None, the feature names will be prefixed with the string
        representation of the `time_lag` parameter followed by an underscore.

    drop_resulting_nan : bool, optional (default=False)
        Whether to drop rows with NaN values resulting from the lag operation.

    """

    def __init__(
        self,
        time_lag: str | pd.Timedelta | dt.timedelta = "1h",
        features_to_lag: str | list[str] = None,
        feature_marker: str = None,
        drop_resulting_nan=False,
    ):
        super().__init__()
        self.time_lag = time_lag
        self.features_to_lag = features_to_lag
        self.feature_marker = feature_marker
        self.drop_resulting_nan = drop_resulting_nan

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_to_lag = (
            [self.features_to_lag]
            if isinstance(self.features_to_lag, str)
            else self.features_to_lag
        )
        self.feature_marker = (
            str(self.time_lag) + "_"
            if self.feature_marker is None
            else self.feature_marker
        )
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["is_fitted_"])
        X = check_and_return_dt_index_df(X)
        if self.features_to_lag is None:
            self.features_to_lag = X.columns
        to_lag = X[self.features_to_lag].copy()
        to_lag.index = to_lag.index + self.time_lag
        to_lag.columns = self.feature_marker + to_lag.columns
        X_transformed = pd.concat([X, to_lag], axis=1)
        if self.drop_resulting_nan:
            X_transformed = X_transformed.dropna()
        return X_transformed


class PdGaussianFilter1D(PdTransformerBC):
    """
    A transformer that applies a 1D Gaussian filter to a Pandas DataFrame.
    The Gaussian filter is a widely used smoothing filter that effectively
    reduces the high-frequency noise in an input signal.

    Parameters
    ----------
    sigma : float, default=5
        Standard deviation of the Gaussian kernel.
        In practice, the value of sigma determines the level of smoothing
        applied to the input signal. A larger value of sigma results in a
         smoother output signal, while a smaller value results in less
          smoothing. However, too large of a sigma value can result in the
           loss of important features or details in the input signal.

    mode : str, default='nearest'
        Points outside the boundaries of the input are filled according to
        the given mode. The default, 'nearest' mode is used to set the values
        beyond the edge of the array equal to the nearest edge value.
        This avoids introducing new values into the smoothed signal that
        could bias the result. Using 'nearest' mode can be particularly useful
        when smoothing a signal with a known range or limits, such as a time
        series with a fixed start and end time.

    truncate : float, default=4.
        The filter will ignore values outside the range
        (mean - truncate * sigma) to (mean + truncate * sigma).
        The truncate parameter is used to define the length of the filter
        kernel, which determines the degree of smoothing applied to the input
        signal.

    Attributes
    ----------
    columns : list
        The column names of the input DataFrame.
    index : pandas.Index
        The index of the input DataFrame.

    Methods
    -------
    get_feature_names_out(input_features=None)
        Get output feature names for the transformed data.
    fit(X, y=None)
        Fit the transformer to the input data.
    transform(X, y=None)
        Transform the input data by applying the 1D Gaussian filter.

    """

    def __init__(self, sigma=5, mode="nearest", truncate=4.0):
        super().__init__()
        self.sigma = sigma
        self.mode = mode
        self.truncate = truncate

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index_ = X.index
        return self

    def transform(self, X: pd.Series | pd.DataFrame, y=None):
        check_is_fitted(self, attributes=["features_"])
        X = check_and_return_dt_index_df(X)
        gauss_filter = partial(
            gaussian_filter1d, sigma=self.sigma, mode=self.mode, truncate=self.truncate
        )

        return X.apply(gauss_filter)


class PdCombineColumns(PdTransformerBC):
    """
    A class that combines multiple columns in a pandas DataFrame using a specified
    function.

    Parameters
    ----------
        columns_to_combine (list or None): A list of column names to combine.
            If None, all columns will be combined.
        function (callable or None): A function or method to apply for combining
            columns.
        function_kwargs (dict or None): Additional keyword arguments to pass to the
            combining function.
        drop_columns (bool): If True, the original columns to combine will be dropped
            from the DataFrame. If False, the original columns will be retained.
        label_name (str): The name of the new column that will store the combined
            values.

    Attributes
    ----------
        columns : list
            The column names of the input DataFrame.
        index : pandas.Index
            The index of the input DataFrame.

    Methods
    -------
        get_feature_names_out(input_features=None)
            Get output feature names for the transformed data.
        fit(X, y=None)
            Fit the transformer to the input data.
        transform(X, y=None)
            Transform the input data by applying the function
    """

    def __init__(
        self,
        columns_to_combine=None,
        function=None,
        function_kwargs=None,
        drop_columns=False,
        label_name="combined",
    ):
        super().__init__()
        self.function_kwargs = function_kwargs
        self.columns_to_combine = columns_to_combine
        self.function = function
        self.drop_columns = drop_columns
        self.label_name = label_name

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.function_kwargs = (
            {} if self.function_kwargs is None else self.function_kwargs
        )
        self.features_ = X.columns
        self.index_ = X.index
        for lab in self.columns_to_combine:
            if lab not in self.features_:
                raise ValueError(f"{lab} is not found in X DataFrame columns")
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["features_"])
        X = check_and_return_dt_index_df(X)
        X_transformed = X.copy()
        if self.drop_columns:
            col_to_return = [
                col for col in self.features_ if col not in self.columns_to_combine
            ]
        else:
            col_to_return = list(self.features_)

        X_transformed[self.label_name] = self.function(
            X_transformed[self.columns_to_combine], **self.function_kwargs
        )

        col_to_return.append(self.label_name)

        return X_transformed[col_to_return]


class PdSTLFilter(PdTransformerBC):
    """
    A transformer that applies Seasonal-Trend decomposition using LOESS (STL)
    to a pandas DataFrame, and filters outliers based on an absolute threshold
    from the residual (error) component of the decomposition.
    Detected outliers are replaced with NaN values.

    Parameters
    ----------
    period : int | str | timedelta
        The periodicity of the seasonal component. Can be specified as:
        - an integer for the number of observations in one seasonal cycle,
        - a string representing the time frequency (e.g., '15T' for 15 minutes),
        - a timedelta object representing the duration of the seasonal cycle.

    trend : int | str | dt.timedelta, optional
        The length of the trend smoother. Must be odd and larger than season
        Statsplot indicate it is usually around 150% of season.
        Strongly depends on your time series.

    absolute_threshold : int | float
        The threshold for detecting anomalies in the residual component.
        Any value in the residual that exceeds this threshold (absolute value)
         is considered an anomaly and replaced by NaN.

    seasonal : int | str | timedelta, optional
        The length of the smoothing window for the seasonal component.
        If not provided, it is inferred based on the period.
        Must be an odd integer if specified as an int.
        Can also be specified as a string representing a time frequency or a
        timedelta object.

    stl_additional_kwargs : dict[str, float], optional
        Additional keyword arguments to pass to the STL decomposition.

    Methods
    -------
    fit(X, y=None)
        Stores the columns and index of the input DataFrame but does not change
        the data. The method is provided for compatibility with the
        scikit-learn pipeline.

    transform(X)
        Applies the STL decomposition to each column of the input DataFrame `X`
        and replaces outliers detected in the residual component with NaN values.
        The outliers are determined based on the provided `absolute_threshold`.

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame with outliers replaced by NaN.
    """

    def __init__(
        self,
        period: int | str | dt.timedelta,
        trend: int | str | dt.timedelta,
        absolute_threshold: int | float,
        seasonal: int | str | dt.timedelta = None,
        stl_additional_kwargs: dict[str, float] = None,
    ):
        super().__init__()
        self.period = period
        self.trend = trend
        self.absolute_threshold = absolute_threshold
        self.seasonal = seasonal
        self.stl_additional_kwargs = stl_additional_kwargs

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.features_ = X.columns
        self.index_ = X.index
        self.stl_ = STLEDetector(
            self.period,
            self.trend,
            self.absolute_threshold,
            self.seasonal,
            self.stl_additional_kwargs,
        )
        self.stl_.fit(X)
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["features_", "stl_"])
        X = check_and_return_dt_index_df(X)
        errors = self.stl_.predict(X)
        errors = errors.astype(bool)
        for col in errors:
            X.loc[errors[col], col] = np.nan

        return X


class PdFillGaps(PdTransformerBC):
    """
    A class designed to identify gaps in time series data and fill them using
    a specified model.

    1- The class identified the gaps to fill and filter them using upper and lower gap
    thresholds.
    2- The biggest group of valid data is identified and is used to fit the model.
    3- The neighboring gaps are filled using backcasting or forecasting.

    The process is repeated at step 2 until there are no more gaps to fill

    Parameters
    ----------
    model_name : str, optional
        The name of the model to be used for filling gaps, by default "STL".
        It must be a key of MODEL_MAP
    model_kwargs : dict, optional
        A dictionary containing the arguments of the model.
    lower_gap_threshold : str or datetime.datetime, optional
        The lower threshold for the size of gaps to be considered, by default None.
    upper_gap_threshold : str or datetime.datetime, optional
        The upper threshold for the size of gaps to be considered, by default None.

    Attributes
    ----------
    model_ : callable
        The predictive model class used to fill gaps, determined by `model_name`.
    features_ : list
        The list of feature columns present in the data.
    index_ : pd.Index
        The index of the data passed during the `fit` method.
    """

    def __init__(
        self,
        model_name: str = "STL",
        model_kwargs: dict = {},
        lower_gap_threshold: str | dt.datetime = None,
        upper_gap_threshold: str | dt.datetime = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.lower_gap_threshold = lower_gap_threshold
        self.upper_gap_threshold = upper_gap_threshold

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.model_ = MODEL_MAP[self.model_name]
        self.features_ = X.columns
        self.index_ = X.index

        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["model_"])
        X = check_and_return_dt_index_df(X)
        gaps = get_data_blocks(
            X,
            is_null=True,
            return_combination=False,
            lower_td_threshold=self.lower_gap_threshold,
            upper_td_threshold=self.upper_gap_threshold,
        )

        for col in X:
            while gaps[col]:
                data_blocks = get_data_blocks(X[col], return_combination=False)[col]
                data_timedelta = [block[-1] - block[0] for block in data_blocks]
                biggest_group = data_blocks[data_timedelta.index(max(data_timedelta))]
                start, end = get_outer_timestamps(biggest_group, X.index)

                indices_to_delete = []
                for i, idx in enumerate(gaps[col]):
                    if start in idx:
                        bc_model = self.model_(backcast=True, **self.model_kwargs)
                        bc_model.fit(X.loc[biggest_group, col])
                        X.loc[idx, col] = (
                            bc_model.predict(idx.to_series()).to_numpy().flatten()
                        )
                        indices_to_delete.append(i)
                    elif end in idx:
                        fc_model = self.model_(**self.model_kwargs)
                        fc_model.fit(X.loc[biggest_group, col])
                        X.loc[idx, col] = (
                            fc_model.predict(idx.to_series()).to_numpy().flatten()
                        )
                        indices_to_delete.append(i)

                for i in sorted(indices_to_delete, reverse=True):
                    del gaps[col][i]

        return X
