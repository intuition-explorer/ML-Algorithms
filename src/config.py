# src/config.py

# Preprocessing
TEST_SIZE = 0.2
RANDOM_STATE = 42
n_pca_components=10
WINDOW_SIZE = 64  # ~500 ms for 128 Hz EEG


# Models
LOG_REG_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "C": 1.0,
    "penalty": "l2"
}

TREE_PARAMS = {
    "max_depth": none,
    "min_samples_leaf": 1,
    "criterion": "entropy",
    "random_state": RANDOM_STATE
}

# Persistent Homology (for later)
PH_PARAMS = {
    "maxdim": 2,
    "max_edge_length": 2.0
}
