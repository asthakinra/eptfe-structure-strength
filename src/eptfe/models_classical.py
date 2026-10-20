"""Classical baselines. Each factory returns a fresh sklearn estimator; imputation,
scaling and monotonic sign learning are part of the pipeline, so they are fit on the
training fold only."""
import numpy as np
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def spearman_signs(X, y):
    """+1 / -1 per column from training-fold rank correlation; zero defaults to +1."""
    return [1 if (stats.spearmanr(X[:, j], y)[0] or 0) >= 0 else -1 for j in range(X.shape[1])]


class MonoRF(RandomForestRegressor):
    def fit(self, X, y, sample_weight=None):
        self.monotonic_cst = spearman_signs(X, y)
        return super().fit(X, y, sample_weight)


class MonoHGB(HistGradientBoostingRegressor):
    def fit(self, X, y, sample_weight=None):
        self.monotonic_cst = spearman_signs(X, y)
        return super().fit(X, y, sample_weight)


def _imp(): return SimpleImputer(strategy='median')


MODELS = {
    'ridge': lambda: make_pipeline(_imp(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 30))),
    'lasso': lambda: make_pipeline(_imp(), StandardScaler(), LassoCV(cv=5, alphas=40, max_iter=20000)),
    'pls3': lambda: make_pipeline(_imp(), StandardScaler(), PLSRegression(3)),
    'rf': lambda: make_pipeline(_imp(), RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.5,
                                                              random_state=0, n_jobs=-1)),
    'mono_rf': lambda: make_pipeline(_imp(), MonoRF(n_estimators=500, min_samples_leaf=2, max_features=0.5,
                                                   random_state=0, n_jobs=-1)),
    'mono_hgb': lambda: make_pipeline(_imp(), MonoHGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=8,
                                                     min_samples_leaf=10, random_state=0)),
    'mlp_sklearn': lambda: make_pipeline(_imp(), StandardScaler(),
                                         MLPRegressor(hidden_layer_sizes=(32,), alpha=1.0, max_iter=3000,
                                                      early_stopping=True, random_state=0)),
}


def fit_predict(name, Xtr, ytr, Xte, log_target=False):
    if name == 'mean':
        return np.full(len(Xte), ytr.mean())
    t = np.log(ytr) if log_target else ytr
    q = np.ravel(MODELS[name]().fit(Xtr, t).predict(Xte))
    return np.exp(q) if log_target else q
