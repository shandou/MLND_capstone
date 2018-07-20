import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    cross_val_score, RandomizedSearchCV, GridSearchCV
)
from sklearn.metrics import roc_auc_score


def timer(method):
    '''
    Timer decorator
    '''
    def timed(*args, **kw):
        t_start = time.time()
        result = method(*args, **kw)
        t_elapsed = time.time() - t_start
        print('{} runtime = {:.2f} s'.format(method.__name__, t_elapsed))
        return result
    return timed


def make_pipe(estimator):
    """Model pipeline"""
    return Pipeline([
        ('scaler', StandardScaler()), ('estimator', estimator)
    ])


def gridsearch(pipe, param_grid, **cv_params):
    """Grid or randomized search for selecting hyperparameters"""
    n_hyperparams = sum(len(v) for v in param_grid.values())
    print(
        'Size of hyperparameter space: n_hyperparams = {}'.format(
            n_hyperparams
        )
    )
    if(n_hyperparams < 5):
        print('Hyperparameter search with GridSearchCV')
        gs = GridSearchCV(pipe, param_grid, verbose=1, **cv_params)
    else:
        print('Hyperparameter search with RandomizedSearchCV')
        gs = RandomizedSearchCV(
            pipe, param_grid, n_iter=3, verbose=1, **cv_params
        )
    return gs


class Classifier():
    '''
    Classifer with grid-search hyperparameter tuning and cross validation
    '''
    global metric
    metric = dict(scoring='roc_auc')

    def __init__(self, gridsearch=True):
        self.gridsearch = gridsearch
        return None

    @timer
    def assess(self, estimator, X, y, param_grid):
        """Performance assessments for selecting algorithms
        Using 5x2 nested cross-validation"""
        pipe = make_pipe(estimator)
        # Inner loop: 2-fold CV for hyperparameter selection
        gs = gridsearch(pipe, param_grid, cv=2, **metric)
        # Outer loop: 5-fold CV for model training
        scores = cross_val_score(gs, X, y, cv=5, **metric)
        return (scores.mean(), scores.std())

    @timer
    def fit(self, estimator, X, y, cv=5, param_grid={}):
        """Train selected model"""
        pipe = make_pipe(estimator)
        if self.gridsearch:
            gs = gridsearch(pipe, param_grid, cv=5, **metric)
            __ = gs.fit(X, y)
            best_estimator = (
                gs.best_estimator_.named_steps['estimator']
            )
            self.pipe = gs.best_estimator_
            if hasattr(best_estimator, 'feature_importances_'):
                self.feature_importances = (
                    best_estimator.feature_importances_
                )
            elif hasattr(best_estimator, 'coef_'):
                self.feature_importances = (
                    best_estimator.coef_
                )
            self.best_params = gs.best_params_
            self.score = gs.best_score_
        else:
            self.pipe = pipe
            if cv is not None:
                scores = cross_val_score(self.pipe, X, y, cv=cv, **metric)
                self.train_score = dict(
                    cv_mean=scores.mean(), cv_std=scores.std()
                )
            __ = pipe.fit(X, y)
            if not cv:
                self.train_score = roc_auc_score(y, pipe.predict(X))
            estimator = pipe.named_steps['estimator']
            if hasattr(estimator, 'feature_importances_'):
                self.feature_importances = estimator.feature_importances_
            elif hasattr(estimator, 'coef_'):
                self.feature_importances = estimator.coef_
        return None

    def predict(self, X):
        """Apply model to new data"""
        return self.pipe.predict(X)