"""Classifiers for Machine Learning experiments.

Exposes five classifiers with the same scikit-learn fit/predict interface:
- Naive Bayes (MultinomialNB for sparse input, GaussianNB for dense input).
- Logistic Regression.
- Random Forest.
- XGBoost.
- Support Vector Machine (LinearSVC).

`make_classifier(name, dense)` selects the correct and pre-configured variant
of each algorithm. Balanced class weights (`class_weight="balanced"`) are used
by default to handle the natural imbalance between ham and smishing.
"""

from __future__ import annotations


CLASSIFIER_NAMES = ["nb", "logreg", "rf", "xgb", "svm"]


def _make_nb(dense: bool):
    if dense:
        from sklearn.naive_bayes import GaussianNB

        return GaussianNB()
    from sklearn.naive_bayes import MultinomialNB

    return MultinomialNB()


def _make_logreg():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        C=1.0,
    )


def _make_rf():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )


def _make_xgb():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="mlogloss",
        random_state=42,
    )

def _make_svm():
    from sklearn.svm import LinearSVC

    return LinearSVC(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )


from typing import Any

def make_classifier(name: str, dense: bool) -> Any:
    """Instantiates and returns a classifier with default hyperparameters."""
    name = name.lower()
    if name == "nb":
        return _make_nb(dense=dense)
    if name == "logreg":
        return _make_logreg()
    if name == "rf":
        return _make_rf()
    if name == "xgb":
        return _make_xgb()
    if name == "svm":
        return _make_svm()
    raise ValueError(f"Unknown classifier: {name}")
