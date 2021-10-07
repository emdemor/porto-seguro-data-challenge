import sklearn
import numpy as np
import pandas as pd
import xgboost

from copy import deepcopy
from itertools import product, combinations_with_replacement
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xtlearn.preprocessing import MinMaxScaler
from xtlearn.feature_selection import FeatureSelector


from base import *


def generate_stratified_folds(train, n_folds=5, shuffle=True, random_state=42):

    temp = train.copy().reset_index(drop=True)

    # Instaciando o estritificador
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    # Gerando os index com os folds
    stratified_folds = list(skf.split(X=temp.drop(columns="y"), y=temp["y"]))

    for fold_index in range(n_folds):

        train_index, validation_index = stratified_folds[fold_index]

        temp.loc[temp[temp.index.isin(validation_index)].index, "fold"] = fold_index

    return temp["fold"].astype(int)


def cross_validate_score(
    X,
    y,
    estimator,
    n_folds=5,
    scoring=f1_score,
    threshold=0.3,
    random_state=42,
    verbose=0,
    fit_params={},
):

    scores = []

    temp = X.assign(y=y)

    temp["fold"] = generate_stratified_folds(temp, n_folds=n_folds)

    iterator = (
        range(n_folds) if verbose < 1 else tqdm(range(n_folds), desc="Cross validation")
    )

    for fold in iterator:

        # Separando os dados de treinamento para essa fold
        train_data = temp[temp["fold"] != fold].copy()

        # Separando os dados de teste para esse fold
        test_data = temp[temp["fold"] == fold].copy()

        X_train = train_data.drop(columns=["fold", "y"]).values

        X_test = test_data.drop(columns=["fold", "y"]).values

        y_train = train_data["y"].values

        y_test = test_data["y"].values

        if estimator.__class__ == xgboost.sklearn.XGBClassifier:
            fit_params["eval_set"] = [(X_test, y_test)]

        try:
            estimator.fit(X_train, y_train, verbose=0, **fit_params)
        except:
            estimator.fit(X_train, y_train, **fit_params)

        prob_test = estimator.predict_proba(X_test)[:, -1]

        scores.append(scoring(y_test, prob_test > threshold))

    avg_score = np.mean(scores)

    return -avg_score


class PolynomialFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, include_bias=True, **kwargs):
        self.kwargs = kwargs
        self.degree = degree
        self.include_bias = include_bias
        self.columns = None

    def fit(self, X, y=None):
        self.columns = X.columns
        self.poly_feat = sklearn.preprocessing.PolynomialFeatures(
            degree=self.degree, include_bias=self.include_bias, **self.kwargs
        ).fit(X, y)
        return self

    def transform(self, X, y=None):

        if self.include_bias:
            cols = ["bias"]
        else:
            cols = []

        for i in range(1, 1 + self.degree):
            cols = cols + [
                "*".join(e)
                for e in list(combinations_with_replacement(self.columns, i))
            ]

        # Dados transformados
        X_transf = pd.DataFrame(
            self.poly_feat.transform(X), columns=cols, index=X.index
        )

        return X_transf

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


# PolynomialFeatures().fit_transform(X.iloc[:, :2])


class Model(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        classifier,
        classifier_params={},
        min_correlation=0.0,
        z_score_range=None,
        polynomial_degree=1,
    ):

        self.classifier = classifier
        self.classifier_params = classifier_params
        self.min_correlation = min_correlation
        self.z_score_range = z_score_range
        self.polynomial_degree = polynomial_degree

        self.set_classifier_params(**classifier_params)
        self.corr_columns = None
        self.selected_columns = None
        self.preprocessing = None

    def fit(self, X, y=None, fit_params={}):

        # Gerando cópias
        X = X.copy()
        y = y.copy()

        # Coletando informações
        self.corr_columns = (
            X.assign(y=y).corr()["y"].drop("y").abs().sort_values(ascending=False)
        )

        # Encontrando as colunas com uma corr mínima
        self.selected_columns = self.corr_columns[
            self.corr_columns > self.min_correlation
        ].index

        # Definindo os limites de zscore
        if self.z_score_range is None:
            min_z, max_z = (-1000, 1000)
        else:
            min_z, max_z = self.z_score_range

        # Pipeline de preprocessamento
        self.preprocessing = Pipeline(
            steps=[
                ("FeatureSelector", FeatureSelector(self.selected_columns)),
                ("ZClipper", ZClipper(min_z=min_z, max_z=max_z)),
                ("Correlatum", Correlatum()),
                ("Imputer", Imputer()),
                ("MinMaxScaler", MinMaxScaler()),
                (
                    "PolynomialFeatures",
                    PolynomialFeatures(
                        degree=self.polynomial_degree, include_bias=False
                    ),
                ),
            ]
        )

        # Treinando o preprocessamento
        self.preprocessing.fit(X, y)

        # Aplicando a transformação
        X = self.transform(X, y)

        # Fitar o classificador
        self.classifier.set_params(**self.classifier_params)

        # Fitar
        self.classifier.fit(X, y, **fit_params)

        return self

    def transform(self, X, y=None):

        X = X.copy()

        if y is not None:
            y = y.copy()

        return self.preprocessing.transform(X)

    def predict(self, X, threshold=0.5):
        X = X.copy()
        X = self.transform(X, y=None)
        predict_prob = self.classifier.predict_proba(X)[:, 1]
        condition = lambda x: x > threshold
        vec_condition = np.vectorize(condition)
        return np.where(vec_condition(predict_prob), 1, 0)

    def predict_proba(self, X):
        X = X.copy()
        X = self.transform(X, y=None)
        return self.classifier.predict_proba(X)[:, 1]

    def set_classifier_params(self, **classifier_params):
        self.classifier.set_params(**classifier_params)

    def cross_validate_score(
        self,
        X,
        y,
        n_folds=5,
        threshold=0.3,
        random_state=42,
        verbose=0,
        fit_params={},
    ):
        X = X.copy()
        y = y.copy()

        self.fit(X, y)

        X = self.transform(X, y)

        return cross_validate_score(
            X=X,
            y=y,
            estimator=self.classifier,
            n_folds=n_folds,
            scoring=f1_score,
            threshold=threshold,
            random_state=random_state,
            verbose=verbose,
            fit_params={},
        )


def new_parameters(parameters):

    new_parameters = deepcopy(parameters)

    for elem in new_parameters:
        if elem["type"] == "real":
            elem["estimate"] = elem["estimate"] + np.random.uniform(
                -elem["step"], elem["step"]
            )
            elem["estimate"] = np.clip(elem["estimate"], *elem["range"])

        elif elem["type"] == "integer":
            elem["estimate"] = elem["estimate"] + int(
                np.round(np.random.uniform(-elem["step"], elem["step"]), 0)
            )
            elem["estimate"] = np.clip(elem["estimate"], *elem["range"])

        elif elem["type"] == "categorical":
            elem["estimate"] = np.random.choice(elem["range"])

    return new_parameters
