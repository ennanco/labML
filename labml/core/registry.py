"""Technique registries for preprocessors, reducers, models and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FastICA, NMF, PCA
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    SelectPercentile,
    f_classif,
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, SVR


@dataclass(frozen=True)
class Technique:
    """Represents one named technique and its estimator builder."""

    name: str
    builder: Callable[..., BaseEstimator | None]


def _none_builder(**_: object) -> None:
    return None


SCALE_REGISTRY: dict[str, Technique] = {
    "none": Technique("none", _none_builder),
    "norm": Technique("norm", lambda **params: Normalizer(**params)),
    "std": Technique("std", lambda **params: StandardScaler(**params)),
}

FILTER_REGISTRY: dict[str, Technique] = {
    "none": Technique("none", _none_builder),
    "fscore": Technique(
        "fscore", lambda **params: SelectPercentile(score_func=f_regression, **params)
    ),
    "mi": Technique(
        "mi",
        lambda **params: SelectPercentile(score_func=mutual_info_regression, **params),
    ),
}

REDUCTION_REGISTRY: dict[str, Technique] = {
    "none": Technique("none", _none_builder),
    "pca": Technique("pca", lambda **params: PCA(**params)),
    "ica": Technique("ica", lambda **params: FastICA(**params)),
    "nmf": Technique("nmf", lambda **params: NMF(**params)),
}

REGRESSION_MODEL_REGISTRY: dict[str, Technique] = {
    "pls": Technique("pls", lambda **params: PLSRegression(**params)),
    "sgd": Technique("sgd", lambda **params: SGDRegressor(**params)),
    "svm": Technique("svm", lambda **params: SVR(**params)),
    "bag": Technique("bag", lambda **params: BaggingRegressor(**params)),
    "gbr": Technique("gbr", lambda **params: GradientBoostingRegressor(**params)),
    "rf": Technique("rf", lambda **params: RandomForestRegressor(**params)),
    "mlp": Technique("mlp", lambda **params: MLPRegressor(**params)),
}

CLASSIFICATION_MODEL_REGISTRY: dict[str, Technique] = {
    "logreg": Technique("logreg", lambda **params: LogisticRegression(**params)),
    "svc": Technique("svc", lambda **params: SVC(**params)),
    "rfc": Technique("rfc", lambda **params: RandomForestClassifier(**params)),
    "bagc": Technique("bagc", lambda **params: BaggingClassifier(**params)),
}

CLASSIFICATION_FILTER_REGISTRY: dict[str, Technique] = {
    "none": Technique("none", _none_builder),
    "fscore": Technique(
        "fscore", lambda **params: SelectPercentile(score_func=f_classif, **params)
    ),
}

DEFAULT_REGRESSION_METRICS = ["neg_root_mean_squared_error", "r2"]
DEFAULT_CLASSIFICATION_METRICS = ["f1_macro", "accuracy"]
