from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

def lr(n_pca_components=10):
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("pca", PCA(n_components=n_pca_components)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    return pipeline


def tree(n_pca_components=10, max_depth=None, min_samples_leaf=1):
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("pca", PCA(n_components=n_pca_components)),
        ("classifier", DecisionTreeClassifier(
            random_state=42,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        ))
    ])
    return pipeline

