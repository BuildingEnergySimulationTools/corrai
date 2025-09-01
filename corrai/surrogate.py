import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

from corrai.base.metrics import nmbe, cv_rmse
from corrai.base.utils import as_1_column_dataframe

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
    "MULTI_LAYER_PERCEPTRON": MLPRegressor(max_iter=5000),
}

GRID_DICT = {
    "TREE_REGRESSOR": [
        {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],  # allow fraction
            "bootstrap": [True, False],
        }
    ],
    "RANDOM_FOREST": [
        {
            "n_estimators": [100, 200, 400, 800],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [1, 2, 5],
        }
    ],
    "LINEAR_REGRESSION": [{"fit_intercept": [True, False]}],
    "LINEAR_SECOND_ORDER": [{"poly__interaction_only": [True, False]}],
    "LINEAR_THIRD_ORDER": [{"poly__interaction_only": [True, False]}],
    "SUPPORT_VECTOR": [
        {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto"],
            "epsilon": [0.001, 0.01, 0.1],
            "degree": [2, 3, 4],  # only relevant if kernel="poly"
        }
    ],
    "MULTI_LAYER_PERCEPTRON": [
        {
            "hidden_layer_sizes": [(50,), (100,), (150,), (50, 50), (100, 50)],
            "activation": [
                "relu",
                "tanh",
            ],  # identity/logistic rarely perform well in regression
            "solver": ["adam", "lbfgs"],  # "sgd" often unstable unless tuned further
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"],
        }
    ],
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
    cv : int, default=10
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
    >>> from corrai.learning.model_selection import MultiModelSO
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
    """

    def __init__(
        self,
        models: list[str] = None,
        cv: int = 10,
        fine_tuning: bool = True,
        scoring: str = "neg_mean_squared_error",
        n_jobs: int = -1,
        random_state: int = None,
    ):
        self.cv = cv
        self.fine_tuning = fine_tuning
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_model_key = None
        self.models = models if models is not None else list(MODEL_MAP.keys())
        self.model_map = {mod: clone(MODEL_MAP[mod]) for mod in self.models}

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series, verbose=True):
        y = y.squeeze() if isinstance(y, pd.DataFrame) else y

        for mod in self.model_map.values():
            mod.fit(X, y)

        score_frame = pd.DataFrame(
            columns=[f"mean({self.scoring})", f"std({self.scoring})"], index=self.models
        )
        for mod_name, mod in self.model_map.items():
            cv_scores = cross_val_score(mod, X, y, scoring=self.scoring, cv=self.cv)
            score_frame.loc[mod_name, f"mean({self.scoring})"] = np.mean(cv_scores)
            score_frame.loc[mod_name, f"std({self.scoring})"] = np.std(cv_scores)

        score_frame.sort_values(f"mean({self.scoring})", ascending=False, inplace=True)

        if verbose:
            print(
                "=== Training results === \n"
                f"Cross validation {self.scoring} scores of {self.cv} folds\n"
                f"{score_frame}"
            )

        self.best_model_key = score_frame.index[0]

        if self.fine_tuning:
            if verbose:
                print(f"\n === Fine tuning === \n" f"Fine tuning {self.best_model_key}")

            self.fine_tune(X, y, self.best_model_key)

        self._is_fitted = True

    def predict(
        self, X: pd.DataFrame | np.ndarray | pd.Series, model: str = None
    ) -> pd.DataFrame:
        check_is_fitted(self)
        model_for_prediction = self.get_model(model)
        if X.ndim == 1:
            X = as_1_column_dataframe(X)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(model_for_prediction.predict(X), index=X.index)
        else:
            return pd.DataFrame(model_for_prediction.predict(X))

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
        verbose=3,
    ):
        if GRID_DICT[model] is None:
            raise ValueError(f"Fine tuning for {model} not yet implemented")

        model_to_tune = self.get_model(model)

        grid_search = GridSearchCV(
            model_to_tune,
            GRID_DICT[model],
            cv=self.cv,
            scoring=self.scoring,
            return_train_score=True,
            verbose=verbose,
            n_jobs=self.n_jobs,
        )
        grid_search.fit(X, y)

        self.model_map[model] = grid_search.best_estimator_
        if verbose > 0:
            cvres = grid_search.cv_results_
            for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
                print(np.sqrt(-mean_score), params)
