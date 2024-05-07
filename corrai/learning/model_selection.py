import enum

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

from corrai.metrics import nmbe, cv_rmse
from corrai.base.utils import as_1_column_dataframe


class Model(str, enum.Enum):
    TREE_REGRESSOR = "TREE_REGRESSOR"
    RANDOM_FOREST = "RANDOM_FOREST"
    LINEAR_REGRESSION = "LINEAR_REGRESSION"
    LINEAR_SECOND_ORDER = "LINEAR_SECOND_ORDER"
    LINEAR_THIRD_ORDER = "LINEAR_THIRD_ORDER"
    SUPPORT_VECTOR = "SUPPORT_VECTOR"
    MULTI_LAYER_PERCEPTRON = "MULTI_LAYER_PERCEPTRON"


MODEL_MAP = {
    Model.TREE_REGRESSOR: RandomForestRegressor(),
    Model.RANDOM_FOREST: RandomForestRegressor(),
    Model.LINEAR_REGRESSION: LinearRegression(),
    Model.LINEAR_SECOND_ORDER: Pipeline(
        [
            ("poly", PolynomialFeatures(2)),
            # Intercept is already added by PolynomialFeatures
            ("Line_reg", LinearRegression(fit_intercept=False)),
        ]
    ),
    Model.LINEAR_THIRD_ORDER: Pipeline(
        [
            ("poly", PolynomialFeatures(3)),
            ("Line_reg", LinearRegression(fit_intercept=False)),
        ]
    ),
    Model.SUPPORT_VECTOR: SVR(),
    Model.MULTI_LAYER_PERCEPTRON: MLPRegressor(max_iter=5000),
}

GRID_DICT = {
    Model.TREE_REGRESSOR: [
        {
            "n_estimators": [100, 200, 500],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
    ],
    Model.RANDOM_FOREST: [
        {"n_estimators": [100, 200, 400, 600]},
    ],
    Model.LINEAR_REGRESSION: [{"fit_intercept": [True, False]}],
    Model.LINEAR_SECOND_ORDER: [{"poly__interaction_only": [True, False]}],
    Model.LINEAR_THIRD_ORDER: [{"poly__interaction_only": [True, False]}],
    Model.SUPPORT_VECTOR: [
        {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": [2, 3, 4],
            "gamma": ["scale", "auto"],
            "epsilon": [0.005, 0.01, 0.1],
        }
    ],
    Model.MULTI_LAYER_PERCEPTRON: [
        {
            "hidden_layer_sizes": [(50,), (100,), (150,)],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": [0.00005, 0.0005, 0.005],
            # 'batch_size': ['auto', 100, 200, 300]
        }
    ],
}


class ModelTrainer:
    def __init__(self, model_pipe, test_size: float = 0.2, random_state: float = 42):
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
        self.model_pipe = model_pipe
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

        self.model_pipe.fit(self.x_train, self.y_train)
        self._is_trained = True

    @property
    def test_nmbe_score(self):
        if self._is_trained:
            return nmbe(self.model_pipe.predict(self.x_test), self.y_test)
        else:
            raise ValueError("Model is not trained yet. use train() method")

    @property
    def test_cvrmse_score(self):
        if self._is_trained:
            return cv_rmse(self.model_pipe.predict(self.x_test), self.y_test)
        else:
            raise ValueError("Model is not trained yet. use train() method")


class MultiModelSO(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        models: list[Model] = Model,
        cv: int = 10,
        fine_tuning: bool = True,
        scoring: str = "neg_mean_squared_error",
        n_jobs: int = -1,
        random_state: int = None,
    ):
        """
        Initialize a MultiModelsSO instance for training and evaluating multiple
        regression models.
        Allows training and comparing multiple regression models,
        selecting the best-performing model, and optionally fine-tuning its
        hyperparameters.

        :param models: A list of regression model names to be trained and
            compared (default: Model Enum).
        :param cv: The number of cross-validation folds (default: 10).
        :param fine_tuning: Whether to perform hyperparameter fine-tuning
            (default: True).
        :param scoring: The scoring metric for model evaluation
            (default: 'neg_mean_squared_error').
        :param n_jobs: The number of CPU cores to use for parallel processing
            (default: -1, all available cores).
        :param random_state: Seed for random number generation (default: None).

        Attributes:
        - model_map: A dictionary mapping model names to model instances.

        Methods:
        - fit(X, y, verbose=True): Train the models on the provided data and select
            the best-performing model.
        - predict(X, model=None): Make predictions using a specified or the best model.
        - get_model(model=None): Retrieve a model instance by name.
        - fine_tune(X, y, model=None, verbose=3): Fine-tune the
            hyperparameters of a model.

        """
        self.cv = cv
        self.fine_tuning = fine_tuning
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_model_key = None
        self.models = models
        self.model_map = {mod: clone(MODEL_MAP[mod]) for mod in self.models}

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series, verbose=True):
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()

        for mod in self.model_map.values():
            mod.fit(X, y)

        score_dict = {}
        for key, mod in self.model_map.items():
            cv_scores = cross_val_score(mod, X, y, scoring=self.scoring, cv=self.cv)

            score_dict[key] = [np.mean(cv_scores), np.std(cv_scores)]
        sorted_score_dict = dict(
            sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        )

        if verbose:
            print(
                "=== Training results ==="
                "Cross validation neg_mean_squared_error scores"
                f"[mean, standard deviation] of {self.cv} folds\n"
            )
            for key, val in sorted_score_dict.items():
                print(f"{key}: {val}")

        self.best_model_key = list(sorted_score_dict)[0]

        if self.fine_tuning:
            if verbose:
                print(f"\n === Fine tuning === \nFine tuning {self.best_model_key}")
            self.fine_tune(X, y, self.best_model_key)

        self._is_fitted = True

        pass

    def predict(
        self, X: pd.DataFrame | np.ndarray | pd.Series, model: Model = None
    ) -> pd.DataFrame:
        check_is_fitted(self)
        model_for_prediction = self.get_model(model)
        if X.ndim == 1:
            X = as_1_column_dataframe(X)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(model_for_prediction.predict(X), index=X.index)
        else:
            return pd.DataFrame(model_for_prediction.predict(X))

    def get_model(self, model: Model = None):
        if model is None:
            check_is_fitted(self)
            model = self.best_model_key

        if model not in self.model_map.keys():
            raise ValueError(f"{model} is not found in 'models' attribute")

        return self.model_map[model]

    def fine_tune(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        model: Model = None,
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
