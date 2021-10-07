import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from copy import deepcopy
from scipy import stats
from scipy.integrate import cumtrapz
from scipy.special import erfinv, erf
from sklearn.base import TransformerMixin, BaseEstimator
from statsmodels.distributions.empirical_distribution import ECDF


def dataframe_transform(data, transformer):
    return pd.DataFrame(
        transformer.transform(data), columns=data.columns, index=data.index
    )


def check_column_types(data, threshold=0.10):
    list_ = []

    for col in data:
        if data[col].dtype == "O":
            list_.append((col, "categorical"))
        else:
            size = len(data[data[col].notna()])

            a = len(data[data[col].notna()][col].unique()) / size

            if a > threshold:
                list_.append((col, "numerical"))
            else:
                list_.append((col, "categorical"))

    return dict(list_)


def optimize_threshold(model, X, y, scoring, range=np.linspace(0.05, 0.95, 10)):
    thresh = 0.0
    best_score = scoring(y, predict(model, X, threshold=thresh))

    for t in range:
        score = scoring(y, predict(model, X, threshold=t))
        if score > best_score:
            best_score = score
            thresh = t

    return thresh


def predict(model, X_test, threshold=0.5):
    predict_prob = model.predict_proba(X_test)[:, 1]
    condition = lambda x: x > threshold
    vec_condition = np.vectorize(condition)
    return np.where(vec_condition(predict_prob), 1.0, 0.0)


def ks_2sample_plot(model, X, y, bins=10):
    y_prob = model.predict_proba(X)[:, 1]

    df = pd.DataFrame({"prob": y_prob, "y": y})
    dist_bad = df.loc[df["y"] == 1, "prob"]
    dist_good = df.loc[df["y"] == 0, "prob"]
    data_bad = dist_bad.sort_values()
    data_good = dist_good.sort_values()
    n_bad = len(data_bad)
    n_good = len(data_good)
    x_bad = np.linspace(0, 1, n_bad)
    x_good = np.linspace(0, 1, n_good)
    cdf_bad = cumtrapz(x=x_bad, y=data_bad)
    cdf_good = cumtrapz(x=x_good, y=data_good)
    cdf_bad = cdf_bad / max(cdf_bad)
    cdf_good = cdf_good / max(cdf_good)
    fig, ax = plt.subplots(ncols=1)
    ax1 = ax.twinx()
    ax.hist(data_bad, histtype="stepfilled", alpha=0.4, density=True, bins=bins)
    ax.hist(data_good, histtype="stepfilled", alpha=0.4, density=True, bins=bins)
    ax1.plot([0] + list(data_bad[1:]) + [1], [0] + list(cdf_bad) + [1], label="Bads")
    ax1.plot([0] + list(data_good[1:]) + [1], [0] + list(cdf_good) + [1], label="Goods")
    ax1.grid(True)
    ax1.legend(loc="right")
    ax1.set_title("Distributions of Bads and Goods (model)")
    ax.set_xlabel("Predict Prob")
    ax.set_ylabel("Probability Desnsity")
    plt.show()

    return stats.ks_2samp(dist_bad, dist_good)

    class Identity(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return X

        def fit_transform(self, X, y=None):
            self.fit(X)
            return self.transform(X)

        def inverse_transform(self, X, y=None):
            return X


class Clip(BaseEstimator, TransformerMixin):
    def __init__(self, range):
        self.range = range

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.clip(X, self.range[0], self.range[1])

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X, y=None):
        return X


class PercetileScaler1D(BaseEstimator, TransformerMixin):
    def __init__(self, unique_percent_threshold=0.1, side="left"):
        self.side = side
        self.unique_percent_threshold = unique_percent_threshold
        self.sample_type = None
        self.sample = None

    def fit(self, X, y=None):

        X_cpy = deepcopy(X)
        X_cpy = X_cpy[~pd.isnull(X_cpy)]

        self.sample = X_cpy
        self.unique_percent = len(np.unique(X_cpy)) / len(X_cpy)

        if self.unique_percent <= self.unique_percent_threshold:
            self.sample_type = "categorical"
            self.side = "mean"
        else:
            self.sample_type = "continuous"

        if self.side == "mean":
            self.ecdf = ECDF(X_cpy, side="right")
            self.ecdf_2 = ECDF(X_cpy, side="left")

        else:
            self.ecdf = ECDF(X, side=self.side)
            self.ecdf_2 = None

        return self

    def transform(self, X, y=None):

        if isinstance(X, list):
            X = np.array(X)

        result = None

        if self.side == "mean":
            result = 0.5 * self.ecdf(X) + 0.5 * self.ecdf_2(X)
        else:
            result = self.ecdf(X)

        return np.where(pd.isnull(X), np.nan, result)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X, y=None):

        temp = pd.Series(X)
        notna = temp[temp.notna()]

        return pd.concat(
            [temp, pd.Series(np.nanquantile(self.sample, notna), index=notna.index)], 1
        )[1].values


class PercetileScaler(BaseEstimator, TransformerMixin):
    def __init__(self, unique_percent_threshold=0.1, side="left"):
        self.side = side
        self.unique_percent_threshold = unique_percent_threshold
        self.sample_type = None

    def fit(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = X

        kwargs = dict(
            unique_percent_threshold=self.unique_percent_threshold, side=self.side
        )

        self.scalers = []

        n_rows, self.n_columns = data.shape

        for i in range(self.n_columns):

            scaler = PercetileScaler1D(**kwargs)

            scaler.fit(data[:, i])

            self.scalers.append(scaler)

        return self

    def transform(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = X

        results = pd.DataFrame()

        for i in range(self.n_columns):
            results[i] = self.scalers[i].transform(data[:, i])

        return results.values

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = X

        results = pd.DataFrame()

        for i in range(self.n_columns):
            results[i] = self.scalers[i].inverse_transform(data[:, i])

        return results.values


class GaussianScaler1D(BaseEstimator, TransformerMixin):
    def __init__(self, unique_percent_threshold=0.1, side="left", mean=0, std=1):
        self.side = side
        self.mean = mean
        self.std = std
        self.unique_percent_threshold = unique_percent_threshold
        self.percentile_scaler = PercetileScaler1D(
            unique_percent_threshold=self.unique_percent_threshold, side=self.side
        )

    def fit(self, X, y=None):
        self.percentile_scaler.fit(X)
        return self

    def transform(self, X, y=None):

        X_temp = self.percentile_scaler.transform(X)

        thresh = 0.00000001

        percentile = np.clip(X_temp, thresh, 1 - thresh)

        # X_temp = np.where(X_temp < thresh, thresh, X_temp)
        # X_temp = np.where(X_temp > (1 - thresh), 1 - thresh, X_temp)

        N = self.__probit(percentile)

        result = self.mean + self.std * N

        return result

    def inverse_transform(self, X, y=None):

        N = (X - self.mean) / self.std

        p = np.clip(self.__probitinv(N), 0, 1)

        result = self.percentile_scaler.inverse_transform(p)

        return result

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def __probit(self, p):
        return np.sqrt(2) * erfinv(2 * p - 1)

    def __probitinv(self, x):
        return 0.5 + 0.5 * erf(x / np.sqrt(2))


class GaussianScaler(BaseEstimator, TransformerMixin):
    def __init__(self, unique_percent_threshold=0.1, side="left", mean=0, std=1):
        self.side = side
        self.mean = mean
        self.std = std
        self.unique_percent_threshold = unique_percent_threshold

    def fit(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = X

        kwargs = dict(
            unique_percent_threshold=self.unique_percent_threshold,
            side=self.side,
            std=self.std,
            mean=self.mean,
        )

        self.scalers = []

        n_rows, self.n_columns = data.shape

        for i in range(self.n_columns):

            scaler = GaussianScaler1D(**kwargs)

            scaler.fit(data[:, i])

            self.scalers.append(scaler)

        return self

    def transform(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = X

        results = pd.DataFrame()

        for i in range(self.n_columns):
            results[i] = self.scalers[i].transform(data[:, i])

        return results.values

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = X

        results = pd.DataFrame()

        for i in range(self.n_columns):
            results[i] = self.scalers[i].inverse_transform(data[:, i])

        return results.values


class WoeCategoricalReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        self.iv_table = None

    def fit(self, X, y):

        X = X.copy()

        y = y.copy()

        categorical_features = X[self.features].assign(y=y)

        self.iv_table = pd.DataFrame()

        self.substitutions = {}

        for col in self.features:

            iv_table = eval_information_value(
                categorical_features[col],
                categorical_features["y"],
                y_values=[0, 1],
                goods=1,
                treat_inf=True,
            ).reset_index()

            iv_table = iv_table.assign(column=col).assign(iv_total=iv_table["iv"].sum())

            self.iv_table = self.iv_table.append(iv_table)

            self.substitutions.update(
                {col: iv_table.set_index("feature")["woe"].to_dict()}
            )

        return self

    def transform(self, X, y=None):
        X = X.copy()

        for col in self.features:
            X[col] = X[col].replace(self.substitutions[col])

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class ZClipper(BaseEstimator, TransformerMixin):
    def __init__(self, min_z=-3, max_z=3, features="all"):
        self.min_z = min_z
        self.max_z = max_z
        self.features = features

    def fit(self, X, y=None):
        self.__min_x = X.mean() + X.std() * self.min_z
        self.__max_x = X.mean() + X.std() * self.max_z
        return self

    def transform(self, X, y=None):
        X = X.copy()

        X = X.clip(lower=self.__min_x, upper=self.__max_x, axis=1)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        return X


def logscale(x):
    return np.sign(x) * np.log(np.abs(x))


def log1pscale(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Correlatum(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        transformations={
            "log_scale": logscale,
            "log_1p_scale": log1pscale,
            # "log": np.log,
            # "expm1": np.expm1,
            "sqrt": lambda x: np.sqrt(np.abs(x)),
            "square": np.square,
            "cube": lambda x: np.power(x, 3),
        },
    ):
        self.transformations = transformations

    def fit(self, X, y):

        # Fazer uma cópia do datafrmae
        X = X.copy()

        # Testar se a entrada é um dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Calcula todos os valores como porcentagens das médias nas colunas
        # X = (X) / np.array([X.mean().values])

        # Calcula a correlação com o target
        corr_base = (
            X.assign(y=y).corr()["y"].drop("y").abs().sort_values(ascending=False)
        )

        # Inicia uma variável para guardar as informações
        self.results = {}

        self.column_transformations = {}

        # Loop pelas colunas
        for col in X.columns:

            # Média da coluna
            # column_mean = X[col].mean()

            # Inicializando a coluna com o valor preprocessado
            result = X[[col]]

            # Assume que a correlação do preprocessamento é a maior
            max_corr = corr_base[col]
            self.results[col] = {col: {"max_corr": max_corr, "best_transf": "none"}}
            self.column_transformations[col] = lambda x: x

            # Calcula o número de infinitos
            n_inf = np.isinf(X[col]).sum()

            # Calcula o número de nans
            n_nan = np.isnan(X[col]).sum()

            # Para cada transformação
            for k in self.transformations:

                # Define a transformação
                transformation = self.transformations[k]

                # Executa a operação
                proposal_transf = transformation(result)

                # Reescalona em termos da média
                proposal_transf = proposal_transf / np.mean(
                    proposal_transf[
                        (~np.isinf(proposal_transf) & (~np.isnan(proposal_transf)))
                    ]
                )

                # Calcula o número de infinitos
                n_inf_transf = np.isinf(proposal_transf.values.flatten()).sum()

                # Calcula o número de nans
                # n_nan_transf = np.isnan(proposal_transf.values.flatten()).sum()

                # Calcula a correlação da transformação com o target
                corr_ = proposal_transf.assign(y=y).corr()["y"].abs()[col]

                # Calcula a razão dessa correlação com a máxima
                ratio = corr_ / max_corr

                # Checa se a transformação gerou alguma melhora
                if (ratio > 1) and (
                    n_inf_transf <= n_inf
                ):  # and (n_nan_transf <= n_nan):

                    max_corr = corr_

                    self.column_transformations[col] = transformation

                    self.results[col] = {col: {"max_corr": max_corr, "best_transf": k}}

        return self

    def transform(self, X, y=None):

        X_ = X.copy()

        for col in X_:
            X_[col] = self.column_transformations[col](X_[col])

        return X_  # .fillna(X_.mean())

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


from sklearn.impute import SimpleImputer


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.imputer = SimpleImputer().fit(X)
        return self

    def transform(self, X, y=None):
        columns = X.columns
        index = X.index
        return pd.DataFrame(self.imputer.transform(X), columns=columns, index=index)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
