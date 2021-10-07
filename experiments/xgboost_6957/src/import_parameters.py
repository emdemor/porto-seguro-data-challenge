import yaml
from sklearn.metrics import f1_score
from src.functions import (
    generate_stratified_folds,
    cross_validate_score,
    generate_space_dimension,
)


hyparams = yaml.safe_load(open("data/hyperparameters.yaml", "r"))

# Hyperparamaters
EARLY_STOPPING_ROUNDS = hyparams["EARLY_STOPPING_ROUNDS"]
EVAL_METRIC = hyparams["EVAL_METRIC"]
VERBOSE = hyparams["VERBOSE"]
RANDOM_STATE = hyparams["RANDOM_STATE"]
N_FOLDS = hyparams["N_FOLDS"]
SCORING = f1_score if hyparams["SCORING"] == "f1_score" else None
THRESHOLD = hyparams["THRESHOLD"]

# Parameters space
parameters = yaml.safe_load(open("data/parameters.yaml", "r"))
space = [generate_space_dimension(x) for x in parameters]
PARAMETER_NAMES = [elem.name for elem in space]
