from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np


space = hp.choice('classifier_type', [
    {
        'preprocessing': hp.choice('RF_preprocessing', ['None', hp.choice('RF_PCA', [0.8, 0.9])]),
        'type': 'RF',
        'min_samples_split': hp.choice('min_samples_split', [2, 4, 7, 10, 12]),
        'max_features': hp.choice('max_features', [0.01, 0.04, 0.08, 0.16, 0.32, 0.64, 0.7, 0.8, 0.9, 0.99]),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    },
    {
        'preprocessing': hp.choice('SVM_Linear_preprocessing', ['None',  hp.choice('SVM_Linear_PCA', [0.8, 0.9])]),
        'type': 'SVM_Linear',
        'C': hp.loguniform('SVM_Linear_C', -15.0, 15.0),
        'penalty': hp.choice('SVM_Linear_Penalty', [
            'l1', 'l2'
        ])
    },
    {
        'preprocessing': hp.choice('SVM_RBF_preprocessing', ['None', hp.choice('SVM_RBF_PCA', [0.8, 0.9])]),
        'type': 'SVM_RBF',
        'C': hp.loguniform('SVM_C', -5, 15.0),
        'gamma': hp.loguniform('SVM_Gamma', -15, 3.0),
        'kernel': 'rbf'
    },
])

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


@ignore_warnings(category=ConvergenceWarning)
def objective(args):
    preprocess = args['preprocessing']
    if preprocess != 'None':
        pca = PCA(n_components=preprocess, svd_solver='full')
        pca.fit(X_train)
        X_train_pc = pca.transform(X_train)
    del args['preprocessing']
    classifier_type = args['type']
    del args['type']
    if classifier_type == 'RF':
        clf = RandomForestClassifier(**args)
    elif classifier_type == 'SVM_Linear':
        params = dict({'dual': False}, **args)
        clf = LinearSVC(**params)
    elif classifier_type == 'SVM_RBF':
        clf = SVC(**args)
    else:
        raise NotImplemented
    if preprocess != 'None':
        clf.fit(X_train_pc, y_train)
    else:
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return (y_pred != y_test).mean()


trials = Trials()


if __name__ == '__main__':
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
    # best = fmin(fn=objective, space=hp.loguniform('x', -5, 15), algo=tpe.suggest, max_evals=100, trials=trials)
    l = trials.losses()
    print(space_eval(space, best))
    x_prime = space_eval(space, best)



