import datetime as dt
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.base import TransformerMixin, BaseEstimator

from corrai.math import time_gradient


class PdTransformerBC(TransformerMixin, BaseEstimator, ABC):
    def __init__(self):
        self.columns = None
        self.index = None

    def get_feature_names_out(self, input_features=None):
        return self.columns

    @abstractmethod
    def fit(self, X, y=None):
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

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
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

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
        return X.dropna(how=self.how)


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

    def __init__(self, new_names):
        super().__init__()
        self.new_names = new_names

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
        X.columns = self.new_names
        self.columns = X.columns
        return X

    def inverse_transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X.columns = self.new_names
        self.columns = X.columns
        return X


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

    def fit(self, X, y=None):
        self.transformer.fit(X)
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
        return pd.DataFrame(
            data=self.transformer.transform(X), index=X.index, columns=X.columns
        )

    def inverse_transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            X.columns = self.columns
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

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
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

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
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

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
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

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
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

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
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

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
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

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
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

    def __init__(self, rule, method=None):
        super().__init__()
        self.rule = rule
        self.method = method
        self.resampled_index = None

    def fit(self, X, y=None):
        self.columns = X.columns
        self.resampled_index = X.resample(self.rule).asfreq()
        return self

    def transform(self, X):
        X = X.apply(pd.to_numeric)
        X_resampled = X.resample(self.rule).agg(self.method)
        self.resampled_index = X_resampled.index
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

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X):
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

    def __init__(self, rule, columns_method, remainder="drop"):
        super().__init__()
        self.rule = rule
        self.columns_method = columns_method
        self.resampled_index = None

        if remainder != "drop" and not callable(remainder):
            raise ValueError(
                "If remainder is no set to drop. Aggregation "
                "method must be provided to ensure transformed "
                "index consistency"
            )
        else:
            self.remainder = remainder
        self._check_columns_method()

    def _check_columns_method(self):
        if not isinstance(self.columns_method, list):
            raise ValueError(
                "Columns_method must be a list of Tuple"
                "first index shall be a list of columns names,"
                "second index shall be an aggregation method"
            )
        for elmt in self.columns_method:
            if not isinstance(elmt[0], list):
                raise ValueError("Tuple first element must be a list" "of columns")
            if not callable(elmt[1]):
                raise ValueError("Tuple second element must be a" "callable")

    def _check_columns(self, X):
        for col_list, _ in self.columns_method:
            for col in col_list:
                if col not in X.columns:
                    raise ValueError("Columns in columns_method not found in" "X")

    def fit(self, X, y=None):
        self._check_columns(X)
        if self.remainder == "drop":
            self.columns = []
            for col_list, _ in self.columns_method:
                self.columns += col_list
        else:
            self.columns = X.columns

        return self

    def transform(self, X):
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

        return transformed_X[self.columns]


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

    features_to_lag : list or None, optional (default=None)
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
        time_lag,
        features_to_lag=None,
        feature_marker=None,
        drop_resulting_nan=False,
    ):
        super().__init__()
        if not isinstance(time_lag, dt.timedelta):
            raise ValueError(
                "Invalid time_lag value. You must provide "
                "a datetime.timedelta object"
            )
        self.time_lag = time_lag
        self.features_to_lag = features_to_lag
        self.drop_resulting_nan = drop_resulting_nan

        if feature_marker is None:
            self.feature_marker = str(time_lag) + "_"
        else:
            if not isinstance(feature_marker, str):
                raise ValueError("feature_marker must be a string")
            self.feature_marker = feature_marker

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        return self

    def transform(self, X, y=None):
        if self.columns is None:
            raise ValueError(
                "Transformer is not fitted yet. Perform fitting using fit() method"
            )

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
        if function_kwargs is None:
            function_kwargs = {}
        self.function_kwargs = function_kwargs
        self.columns_to_combine = columns_to_combine
        self.function = function
        self.drop_columns = drop_columns
        self.label_name = label_name

    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        for lab in self.columns_to_combine:
            if lab not in self.columns:
                raise ValueError(f"{lab} is not found in X DataFrame columns")
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if self.drop_columns:
            col_to_return = [
                col for col in self.columns if col not in self.columns_to_combine
            ]
        else:
            col_to_return = list(self.columns)

        X_transformed[self.label_name] = self.function(
            X_transformed[self.columns_to_combine], **self.function_kwargs
        )

        col_to_return.append(self.label_name)

        return X_transformed[col_to_return]


class PdAddFourierPairs(PdTransformerBC):
    """
    A pandas transformer that adds a pair of new columns with sine and cosine
     signal of given frequency and lag.
     Based on time series index, phase shift is computed from the beginning
     of the year.

    Parameters:
    -----------
    frequency : float | int
        The frequency of the sine signal (Hz).
    amplitude : float | int, optional
        Amplitude of the sine signal (default is 1.0).
    feature_marker : str, optional
        The name of the new feature column (default is f"Sine_f_{frequency}").

    Methods:
    --------
    fit(X, y=None)
        Fit method (does nothing, present for compatibility).
    transform(X)
        Transform method that adds a sine signal column to the input DataFrame.

    Returns:
    --------
    X : pd.DataFrame
        Transformed DataFrame with the new sine signal column.
    """

    def __init__(
        self,
        frequency: float | int,
        amplitude: float | int = None,
    ):
        super().__init__()
        self.frequency = frequency

        if amplitude is None:
            self.amplitude = 1.0
        else:
            self.amplitude = amplitude

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        begin = X.index[0]
        year_start = pd.Timestamp(begin.year, 1, 1)
        if begin.tz:
            year_start = year_start.tz_localize(begin.tz)
        seconds_from_start_of_year = (begin - year_start).total_seconds()
        phi = 2 * np.pi * self.frequency * seconds_from_start_of_year

        new_index = X.index.to_frame().diff().squeeze()
        sec_dt = [element.total_seconds() for element in new_index]
        increasing_seconds = pd.Series(sec_dt).cumsum().to_numpy()
        increasing_seconds[0] = 0
        X[f"{self.frequency}_f_Sine"] = self.amplitude * np.sin(
            2 * np.pi * self.frequency * increasing_seconds + phi
        )
        X[f"{self.frequency}_f_Cosine"] = self.amplitude * np.cos(
            2 * np.pi * self.frequency * increasing_seconds + phi
        )

        return X
