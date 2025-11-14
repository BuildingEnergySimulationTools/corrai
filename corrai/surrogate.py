import time

import numpy as np
import pandas as pd

from scipy.stats import loguniform, randint

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

from corrai.base.metrics import nmbe, cv_rmse
from corrai.base.utils import as_1_column_dataframe
from corrai.base.model import Model

GRID_DICT = {
    "TREE_REGRESSOR": [
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True],
        }
    ],
    "RANDOM_FOREST": [
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [1, 2, 5],
            "min_samples_split": [2, 5],
        }
    ],
    "LINEAR_REGRESSION": [{"fit_intercept": [True, False]}],
    "LINEAR_SECOND_ORDER": [{"poly__interaction_only": [True, False]}],
    "LINEAR_THIRD_ORDER": [{"poly__interaction_only": [True, False]}],
    "SUPPORT_VECTOR": [
        {
            "kernel": ["rbf", "linear"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01],
            "epsilon": [0.01, 0.1],
        }
    ],
    "MULTI_LAYER_PERCEPTRON": [
        {
            "MLP__hidden_layer_sizes": [
                (100,),
                (150,),
                (100, 50),
                (150, 100),
            ],
            "MLP__activation": ["relu", "tanh"],
            "MLP__solver": ["adam"],
            "MLP__alpha": [0.0001, 0.001, 0.01],
            "MLP__learning_rate": ["adaptive"],
            "MLP__max_iter": [3000, 5000],
            "MLP__early_stopping": [True],
            "MLP__validation_fraction": [0.15],
            "MLP__n_iter_no_change": [15],
            "MLP__tol": [1e-4],
        }
    ],
}

# For use with RandomizedSearchCV - continuous distributions
# This allows exploring MORE distinct values
GRID_DICT_CONTINUOUS = {
    "TREE_REGRESSOR": {
        "n_estimators": randint(100, 400),
        "max_depth": [10, 20, 30, 40, None],
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True],
    },
    "RANDOM_FOREST": {
        "n_estimators": randint(100, 400),
        "max_depth": [10, 20, 30, 40, None],
        "max_features": ["sqrt", "log2"],
        "min_samples_leaf": randint(1, 10),
        "min_samples_split": randint(2, 15),
    },
    "LINEAR_REGRESSION": {"fit_intercept": [True, False]},
    "LINEAR_SECOND_ORDER": {"poly__interaction_only": [True, False]},
    "LINEAR_THIRD_ORDER": {"poly__interaction_only": [True, False]},
    "SUPPORT_VECTOR": {
        "kernel": ["rbf", "linear"],
        "C": loguniform(0.1, 100),  # Continuous log scale
        "gamma": ["scale", "auto"]
        + list(loguniform(1e-4, 1e-1).rvs(5, random_state=42)),
        "epsilon": loguniform(0.001, 0.5),
    },
    "MULTI_LAYER_PERCEPTRON": {
        "MLP__hidden_layer_sizes": [
            (50,),
            (100,),
            (150,),
            (200,),
            (50, 50),
            (100, 50),
            (150, 100),
            (200, 100),
            (100, 50, 25),
            # (200, 50, 25),
        ],
        "MLP__activation": ["relu", "tanh"],
        "MLP__solver": ["adam", "lbfgs"],
        "MLP__alpha": loguniform(1e-5, 1e-2),
        "MLP__learning_rate": ["constant", "adaptive"],
        "MLP__max_iter": [3000, 5000],
        "MLP__early_stopping": [True],
        "MLP__validation_fraction": [0.05, 0.1],
        "MLP__n_iter_no_change": [15, 30, 50],
        "MLP__tol": [1e-4, 1e-5],
    },
}


# RECOMMENDED: Different n_iter for different model complexities
TUNING_N_ITER_BY_MODEL = {
    "TREE_REGRESSOR": 30,
    "RANDOM_FOREST": 30,
    "LINEAR_REGRESSION": 2,
    "LINEAR_SECOND_ORDER": 2,
    "LINEAR_THIRD_ORDER": 2,
    "SUPPORT_VECTOR": 25,
    "MULTI_LAYER_PERCEPTRON": 20,
}


MODEL_MAP = {
    "TREE_REGRESSOR": RandomForestRegressor(),
    "RANDOM_FOREST": RandomForestRegressor(),
    "LINEAR_REGRESSION": LinearRegression(),
    "LINEAR_SECOND_ORDER": Pipeline(
        [
            ("poly", PolynomialFeatures(2)),
            # Intercept is already added by PolynomialFeatures
            ("Line_reg", LinearRegression(fit_intercept=False)),
        ]
    ),
    "LINEAR_THIRD_ORDER": Pipeline(
        [
            ("poly", PolynomialFeatures(3)),
            ("Line_reg", LinearRegression(fit_intercept=False)),
        ]
    ),
    "SUPPORT_VECTOR": SVR(),
    "MULTI_LAYER_PERCEPTRON": Pipeline(
        [("scaler", StandardScaler()), ("MLP", MLPRegressor(max_iter=5000))]
    ),
}


class ModelTrainer:
    def __init__(self, model, test_size: float = 0.2, random_state: float = 42):
        """
        Initialize a ModelTrainer instance for training a machine learning model.

        :param model_pipe: A scikit-learn compatible model pipeline for training
            and prediction.
        :param test_size: The proportion of the dataset to set aside as the test
            set (default: 0.2).
        :param random_state: Seed for random number generation to ensure
            reproducibility (default: 42).

        The ModelTrainer prepares data for training and evaluation of the
        specified model.

        Attributes:
        - test_size: The proportion of data to be used as the test set.
        - model_pipe: The machine learning model pipeline to be trained.
        - random_state: Seed for random number generation.
        - x_train: Training data features.
        - x_test: Test data features.
        - y_train: Training data labels.
        - y_test: Test data labels.
        - _is_trained: A boolean indicating if the model has been trained.
        """
        self.test_size = test_size
        self.model = model
        self.random_state = random_state
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self._is_trained = False

    def train(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        if isinstance(y, pd.Series):
            y = as_1_column_dataframe(y)

        (self.x_train, self.x_test, self.y_train, self.y_test) = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.model.fit(self.x_train, self.y_train)
        self._is_trained = True

    @property
    def test_nmbe_score(self):
        if self._is_trained:
            return nmbe(self.model.predict(self.x_test), self.y_test)
        else:
            raise ValueError("Model is not trained yet. use train() method")

    @property
    def test_cvrmse_score(self):
        if self._is_trained:
            return cv_rmse(self.model.predict(self.x_test), self.y_test)
        else:
            raise ValueError("Model is not trained yet. use train() method")


class MultiModelSO(BaseEstimator, RegressorMixin):
    """
    Multi-model selection and optimization wrapper for scikit-learn regressors.

    This class automates model training, cross-validation scoring,
    model selection, and optional fine-tuning via grid search.
    It compares multiple candidate models and selects the one with the
    best cross-validation performance according to a specified scoring metric.

    Parameters
    ----------
    models : list of str, optional
        List of model keys to evaluate. Must be a subset of ``MODEL_MAP``.
            "TREE_REGRESSOR", "RANDOM_FOREST", "LINEAR_REGRESSION",
            "LINEAR_SECOND_ORDER", "LINEAR_THIRD_ORDER", "SUPPORT_VECTOR",
            "MULTI_LAYER_PERCEPTRON"
        If ``None`` (default), all models in ``MODEL_MAP`` are evaluated.
    cv : int, default=3
        Number of cross-validation folds to use for model comparison.
    fine_tuning : bool, default=True
        If True, perform a grid search on the best model to fine-tune its
        hyperparameters.
    scoring : str, default="neg_mean_squared_error"
        Scoring function to evaluate models. Should be a valid scikit-learn
        scorer string (e.g. ``"r2"``, ``"neg_mean_absolute_error"``).
    n_jobs : int, default=-1
        Number of parallel jobs for cross-validation and grid search.
        ``-1`` means using all available cores.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    model_map : dict
        Dictionary mapping model keys to fitted estimator instances.
    best_model_key : str
        The key of the best-performing model after training.
    _is_fitted : bool
        Whether the estimator has been fitted.

    Methods
    -------
    fit(X, y, verbose=True)
        Train and evaluate all models, selecting the best one.
    predict(X, model=None)
        Predict using the best model (or a specified model).
    get_model(model=None)
        Retrieve the fitted estimator by key.
    fine_tune(X, y, model=None, verbose=3)
        Perform grid search hyperparameter tuning on a given model.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from corrai.surrogate import MultiModelSO
    >>>
    >>> data = load_diabetes(as_frame=True)
    >>> X = data.data
    >>> y = data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=42
    ... )
    >>>
    >>> model = MultiModelSO(
    ...     models=["LINEAR_REGRESSION", "RANDOM_FOREST"], cv=5, fine_tuning=False
    ... )
    >>> model.fit(X_train, y_train, verbose=True)
    === Training results ===
    Cross validation neg_mean_squared_error scores of 5 folds
                                  mean(neg_mean_squared_error) std(neg_mean_squared_error)
    RANDOM_FOREST                                -3143.015307                          355.466814
    LINEAR_REGRESSION                            -3425.368758                          525.460964
    >>> y_pred = model.predict(X_test)
    >>> y_pred.head()
          0
    287  139.547558
    211  179.517208
    72   134.038756
    321  291.417029
    73   123.789659

    >>> # Fast configuration and training (development)
    >>> model = MultiModelSO(
    ...     models=["LINEAR_REGRESSION", "RANDOM_FOREST", "MULTI_LAYER_PERCEPTRON"],
    ...     cv=3,
    ...     fine_tuning=True,
    ...     tuning_n_iter=TUNING_N_ITER_BY_MODEL,
    ...     use_continuous_distributions=False,
    ...     n_jobs=-1,
    ... )

    >>> # Optimal configuration (production)
    >>> model = MultiModelSO(
    ...     models=None,
    ...     cv=5,
    ...     fine_tuning=True,
    ...     tuning_n_iter=TUNING_N_ITER_BY_MODEL,
    ...     use_continuous_distributions=True,
    ...     n_jobs=-1,
    ...     random_state=42,
    ... )
    """

    def __init__(
        self,
        models: list[str] = None,
        cv: int = 3,
        scoring: str = "neg_mean_squared_error",
        fine_tuning: bool = True,
        tuning_n_iter: int | dict = None,
        use_continuous_distributions: bool = False,
        n_jobs: int = -1,
        random_state: int = None,
    ):
        self.cv = cv
        self.fine_tuning = fine_tuning
        self.tuning_n_iter = tuning_n_iter
        self.use_continuous_distributions = use_continuous_distributions
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_model_key = None
        self.models = models if models is not None else list(MODEL_MAP.keys())
        self.model_map = {mod: clone(MODEL_MAP[mod]) for mod in self.models}

        self.training_times_ = {}
        self.cv_scores_ = {}

    @property
    def feature_names_in_(self):
        check_is_fitted(self, ["_is_fitted"])
        return self.get_model().feature_names_in_

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_is_fitted") and self._is_fitted

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series, verbose=True):
        y = y.squeeze() if isinstance(y, pd.DataFrame) else y

        # Train all models with timing
        for mod_name, mod in self.model_map.items():
            start_time = time.time()
            try:
                mod.fit(X, y)
                self.training_times_[mod_name] = time.time() - start_time
            except Exception as e:
                if verbose:
                    print(f"Warning: {mod_name} failed to train: {e}")
                self.training_times_[mod_name] = None

        # Cross-validation scores
        score_frame = pd.DataFrame(
            columns=[
                f"mean({self.scoring})",
                f"std({self.scoring})",
                "train_time_sec",
            ],
            index=self.models,
        )

        for mod_name, mod in self.model_map.items():
            if self.training_times_[mod_name] is None:
                continue

            cv_scores = cross_val_score(
                mod,
                X,
                y,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,  # NEW: parallel CV
            )

            score_frame.loc[mod_name, f"mean({self.scoring})"] = np.mean(cv_scores)
            score_frame.loc[mod_name, f"std({self.scoring})"] = np.std(cv_scores)
            score_frame.loc[mod_name, "train_time_sec"] = self.training_times_[mod_name]

            self.cv_scores_[mod_name] = cv_scores

        # Remove failed models
        score_frame = score_frame.dropna()
        score_frame.sort_values(f"mean({self.scoring})", ascending=False, inplace=True)

        if verbose:
            print(
                "=== Training results === \n"
                f"Cross validation {self.scoring} scores of {self.cv} folds\n"
                f"{score_frame}"
            )

        if len(score_frame) == 0:
            raise ValueError("All models failed to train!")

        self.best_model_key = score_frame.index[0]

        if self.fine_tuning:
            if verbose:
                print(f"\n === Fine tuning === \n" f"Fine tuning {self.best_model_key}")

            self.fine_tune(X, y, self.best_model_key, verbose=verbose)

        self.target_name_ = y.name
        self._is_fitted = True

        return self

    def predict(
        self, X: pd.DataFrame | np.ndarray | pd.Series, model: str = None
    ) -> pd.DataFrame:
        check_is_fitted(self)
        model_for_prediction = self.get_model(model)
        if X.ndim == 1:
            X = as_1_column_dataframe(X)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                model_for_prediction.predict(X),
                index=X.index,
                columns=[self.target_name_],
            )
        else:
            return pd.DataFrame(
                model_for_prediction.predict(X), columns=[self.target_name_]
            )

    def get_model(self, model: str = None):
        if model is None:
            model = self.best_model_key

        if model not in self.model_map.keys():
            raise ValueError(f"{model} is not found in 'models' attribute")

        return self.model_map[model]

    def fine_tune(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        model: str = None,
        verbose=True,
    ):
        if model is None:
            model = self.best_model_key

        if self.use_continuous_distributions:
            param_dist = GRID_DICT_CONTINUOUS.get(model)
        else:
            param_dist = GRID_DICT.get(model)

        if param_dist is None:
            if verbose:
                print(f"Fine tuning for {model} not available - skipping")
            return

        model_to_tune = self.get_model(model)

        if isinstance(self.tuning_n_iter, dict):
            n_iter = self.tuning_n_iter.get(model, 20)
        elif self.tuning_n_iter is not None:
            n_iter = self.tuning_n_iter
        else:
            n_iter = TUNING_N_ITER_BY_MODEL.get(model, 20)

        verbose_level = 2 if verbose else 0

        grid_search = RandomizedSearchCV(
            model_to_tune,
            param_distributions=param_dist,
            cv=self.cv,
            n_iter=n_iter,
            scoring=self.scoring,
            return_train_score=True,
            verbose=verbose_level,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        start_time = time.time()
        grid_search.fit(X, y)
        tuning_time = time.time() - start_time

        self.model_map[model] = grid_search.best_estimator_

        if verbose:
            print(f"\nFine tuning completed in {tuning_time:.2f}s")
            print(f"Best params: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")

            # Show top 5 configurations
            cvres = grid_search.cv_results_
            results = sorted(
                zip(cvres["mean_test_score"], cvres["params"]), reverse=True
            )[:5]

            print("\nTop 5 configurations:")
            for i, (score, params) in enumerate(results, 1):
                print(f"{i}. Score: {score:.4f} | Params: {params}")

    def get_feature_importance(self, model: str = None, top_n: int = 10):
        check_is_fitted(self)
        mod = self.get_model(model)

        # Handle Pipeline
        if isinstance(mod, Pipeline):
            mod = mod.steps[-1][1]

        if not hasattr(mod, "feature_importances_"):
            raise ValueError(
                f"Model {model or self.best_model_key} doesn't support feature_importances_"
            )

        importance_df = pd.DataFrame(
            {"feature": self.feature_names_in_, "importance": mod.feature_importances_}
        ).sort_values("importance", ascending=False)

        return importance_df.head(top_n) if top_n else importance_df


class StaticScikitModel(Model):
    """
    Wrapper class for static surrogate MultiModelSingleOutput class and scikit-learn
    regressors within the Corrai framework.

    This class adapts corrai's `MultiModelSO` and scikit-learn models
    to the :class:`Model` interface, enabling parameter-to-property mapping and
    simulation execution. It is intended for non-dynamic (static) models where
    outputs are single values or vectors rather than time-dependent series.

    Parameters
    ----------
    scikit_model : MultiModelSO or RegressorMixin
        The underlying scikit-learn model or a Corrai
        :class:`MultiModelSO` meta-estimator.
    target_name : str, optional
        Name of the output variable. Required when ``scikit_model``
        is not an instance of :class:`MultiModelSO`.

    Attributes
    ----------
    is_dynamic : bool
        Always ``False`` for this wrapper, since it represents static models.
    scikit_model : MultiModelSO or RegressorMixin
        The wrapped scikit-learn model used for predictions.
    target_name : str
        Output variable name.

    Raises
    ------
    ValueError
        If ``target_name`` cannot be inferred and is not provided.
    """

    def __init__(
        self, scikit_model: MultiModelSO | RegressorMixin, target_name: str = None
    ):
        super().__init__(is_dynamic=False)
        self.scikit_model = scikit_model
        self.target_name = self._resolve_target_name(target_name)

    def _resolve_target_name(self, target_name: str = None) -> str:
        if target_name is not None:
            return target_name

        if hasattr(self.scikit_model, "target_name_"):
            return self.scikit_model.target_name_

        raise ValueError(
            "target_name must be specified when scikit_model "
            "is not an instance of MultiModelSO"
        )

    def _build_feature_dataframe(
        self, property_dict: dict[str, str | int | float], simulation_options: dict
    ) -> pd.DataFrame:
        if not property_dict and not simulation_options:
            return pd.DataFrame()

        # Merge dictionaries (simulation_options override property_dict)
        merged = {**(property_dict or {}), **(simulation_options or {})}
        return pd.DataFrame([merged])

    def _validate_features(self, features: pd.DataFrame) -> None:
        unknown = set(features.columns) - set(self.scikit_model.feature_names_in_)
        if unknown:
            raise ValueError(f"Unknown features: {unknown}")

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        **simulation_kwargs,
    ) -> pd.Series:
        """
        Run the scikit-learn model prediction.

        Combines provided parameter values and simulation options into a
        feature vector, validates compatibility with the underlying model,
        and returns predictions as a pandas Series.

        Parameters
        ----------
        property_dict : dict of {str: int, float, or str}, optional
            Mapping from feature names to values to use for prediction.
        simulation_options : dict, optional
            Additional feature overrides or configuration parameters to
            include in the feature vector. These values override those
            in ``property_dict`` if keys overlap.
        **simulation_kwargs
            Extra keyword arguments for future extensions (currently unused).

        Returns
        -------
        pd.Series
            Prediction results with index ``[self.target_name]``.

        Raises
        ------
        ValueError
            If unknown feature names are provided.
        """
        features = self._build_feature_dataframe(property_dict, simulation_options)
        self._validate_features(features)

        pred = self.scikit_model.predict(features)
        if isinstance(pred, pd.DataFrame):
            pred = pred.squeeze()

        return pd.Series(pred, index=[self.target_name])
