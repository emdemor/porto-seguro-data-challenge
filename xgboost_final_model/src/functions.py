import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, make_scorer
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import StratifiedKFold


def update_parameter_yaml(results, parameters, filename="data/parameters.yaml"):

    best_par = results.iloc[0].to_dict()

    for dict_ in parameters:

        if dict_["type"] == "real":
            dict_["estimate"] = float(best_par[dict_["parameter"]])

        elif dict_["type"] == "integer":
            dict_["estimate"] = int(best_par[dict_["parameter"]])

    with open(filename, "w") as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)


def generate_space_dimension(dict_par):

    # min_value = max(dict_par["estimate"] - dict_par["step"], dict_par["range"][0])
    # max_value = min(dict_par["estimate"] + dict_par["step"], dict_par["range"][1])

    min_value = dict_par["range"][0]
    max_value = dict_par["range"][1]

    if dict_par["type"] == "real":

        return Real(
            low=min_value,
            high=max_value,
            prior=dict_par["scale"],
            name=dict_par["parameter"],
        )

    elif dict_par["type"] == "integer":

        return Integer(
            low=int(round(min_value)),
            high=int(round(max_value)),
            prior=dict_par["scale"],
            name=dict_par["parameter"],
        )


def generate_stratified_folds(train, n_folds=5, shuffle=True, random_state=42):

    temp = train.copy()

    # Instaciando o estritificador
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    # Gerando os index com os folds
    stratified_folds = list(skf.split(X=temp.drop(columns="y"), y=temp["y"]))

    for fold_index in range(n_folds):

        train_index, validation_index = stratified_folds[fold_index]

        temp.loc[train[temp.index.isin(validation_index)].index, "fold"] = fold_index

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

    # estimator = XGBClassifier(
    #    n_jobs=n_jobs,
    #    random_state=random_state,
    # )

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

        fit_params["eval_set"] = [(X_test, y_test)]

        estimator.fit(X_train, y_train, **fit_params)

        prob_test = estimator.predict_proba(X_test)[:, -1]

        scores.append(scoring(y_test, prob_test > threshold))

    avg_score = np.mean(scores)

    return -avg_score


from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize


# EARLY_STOPPING_ROUNDS = 100
# EVAL_METRIC = "auc"
# VERBOSE = 0
# RANDOM_STATE = 42
# N_FOLDS = 5
# SCORING = f1_score
# THRESHOLD = 0.35

# steps = {
#     "learning_rate": 0.001,
#     "n_estimators": 1,
#     "max_depth": 1,
#     "subsample": 0.001,
#     "min_child_weight": 0.01,
#     "reg_alpha": 0.001,
#     "reg_lambda": 0.001,
#     "colsample_bynode": 0.001,
#     "colsample_bytree": 0.001,
#     "num_parallel_tree": 1,
#     "max_delta_step": 1,
# }

# # Importando resultados ja obtidos
# results = pd.read_csv("data/optimization.csv").sort_values("score")
# x0 = list(results.drop(["iteration", "score"], 1).iloc[0].values)
# y0 = results["score"].iloc[0]
# par_0 = dict(zip(results.drop(["iteration", "score"], 1).columns, x0))

# # Definindo o space
# space = [
#     Real(
#         par_0["learning_rate"] - steps["learning_rate"],
#         par_0["learning_rate"] + steps["learning_rate"],
#         "log-uniform",
#         name="learning_rate",
#     ),
#     Integer(
#         int(par_0["n_estimators"]) - steps["n_estimators"],
#         int(par_0["n_estimators"]) + steps["n_estimators"],
#         "log-uniform",
#         name="n_estimators",
#     ),
#     Integer(
#         int(par_0["max_depth"]),
#         int(par_0["max_depth"] + steps["max_depth"]),
#         name="max_depth",
#     ),
#     Real(
#         par_0["subsample"] - steps["subsample"],
#         par_0["subsample"] + steps["subsample"],
#         name="subsample",
#     ),
#     Real(
#         par_0["min_child_weight"] - steps["min_child_weight"],
#         par_0["min_child_weight"] + steps["min_child_weight"],
#         name="min_child_weight",
#     ),
#     Real(
#         par_0["reg_alpha"] - steps["reg_alpha"],
#         par_0["reg_alpha"] + steps["reg_alpha"],
#         name="reg_alpha",
#     ),
#     Real(
#         par_0["reg_lambda"] - steps["reg_lambda"],
#         par_0["reg_lambda"] + steps["reg_lambda"],
#         name="reg_lambda",
#     ),
#     Real(
#         par_0["colsample_bynode"] - steps["colsample_bynode"],
#         par_0["colsample_bynode"] + steps["colsample_bynode"],
#         name="colsample_bynode",
#     ),
#     Real(
#         par_0["colsample_bytree"] - steps["colsample_bytree"],
#         par_0["colsample_bytree"] + steps["colsample_bytree"],
#         name="colsample_bytree",
#     ),
#     Integer(
#         int(par_0["num_parallel_tree"]),
#         int(par_0["num_parallel_tree"] + steps["num_parallel_tree"]),
#         name="num_parallel_tree",
#     ),
#     Integer(
#         int(par_0["max_delta_step"]),
#         int(par_0["max_delta_step"] + steps["max_delta_step"]),
#         name="max_delta_step",
#     ),
# ]


# PARAMETER_NAMES = [elem.name for elem in space]


# @use_named_args(space)
# def train_function(**params):

#     print(f"Testing parameters: {params.values()}")

#     #     estimator = XGBClassifier(
#     #         n_jobs=n_jobs, random_state=RANDOM_STATE, **dict(zip(PARAMETER_NAMES, params))
#     #     )
#     estimator = XGBClassifier(n_jobs=n_jobs, random_state=RANDOM_STATE, **params)

#     fit_params = {
#         "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
#         "eval_metric": EVAL_METRIC,
#         "verbose": VERBOSE,
#     }

#     return cross_validate_score(
#         X=train.drop("y", 1),
#         y=train["y"],
#         estimator=estimator,
#         fit_params=fit_params,
#         n_folds=N_FOLDS,
#         scoring=SCORING,
#         threshold=THRESHOLD,
#         random_state=RANDOM_STATE,
#         verbose=0,
#     )
