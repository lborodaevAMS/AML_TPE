from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval, rand
from hyperopt.fmin import generate_trials_to_calculate
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial


pd.options.display.max_rows = 999
pd.options.display.max_columns = 50

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



@ignore_warnings(category=ConvergenceWarning)
def objective(args, data):
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    preprocess = args['preprocessing']
    if preprocess != 'None':
        pca = PCA(n_components=preprocess, svd_solver='full')
        pca.fit(X_train)
        X_train_pc = pca.transform(X_train)
        X_test_pc = pca.transform(X_test)
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
        y_pred = clf.predict(X_test_pc)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    del clf
    return (y_pred != y_test).mean()


def plot_losses(losses):
    best_so_far = np.inf
    x = np.full(len(losses) // 5, np.inf)
    for i in range(len(losses)):
        if losses[i] < best_so_far:
            best_so_far = losses[i]
        if i % 5 == 0:
            x[i // 5] = best_so_far
    plt.plot(range(len(x)), x, marker='x')
    plt.show()


def precalculate_performance():
    df = pickle.load(open('metafeatures_original_perf.p', 'rb'))
    for c in ['RF_PCA', 'RF_preprocessing', 'SVM_C', 'SVM_Gamma', 'SVM_Linear_C', 'SVM_Linear_PCA', 'SVM_Linear_Penalty',
                'SVM_Linear_preprocessing', 'SVM_RBF_PCA', 'SVM_RBF_preprocessing', 'classifier_type', 'criterion', 'max_features',
                'min_samples_split']:
        df[c] = np.nan
    for idx, row in df.iterrows():
        print('Processing {} out of {}...'.format(idx, len(df)))
        path_prefix = 'original_datasets/' + row['dataset_name']
        try:
            X_train = pickle.load(open(path_prefix + '/X_train.p', 'rb'))
            X_test = pickle.load(open(path_prefix + '/X_test.p', 'rb'))
            y_train = pickle.load(open(path_prefix + '/y_train.p', 'rb'))
            y_test = pickle.load(open(path_prefix + '/y_test.p', 'rb'))
            assert X_train.shape[1] == X_test.shape[1]
        except FileNotFoundError:
            print(path_prefix + ' not found')
        trials = Trials()
        data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
        fmin_objective = partial(objective, data=data)
        best = fmin(fn=fmin_objective, space=space, algo=rand.suggest, max_evals=50, trials=trials)
        for k, v in trials.best_trial['misc']['idxs'].items():
            if len(v) > 0:
                row.at[k] = v[0]
        for k, v in trials.best_trial['misc']['vals'].items():
            if len(v) > 0:
                row.at[k] = v[0]
    pickle.dump(df, open('metafeatures_original_perf_new.p', 'wb'))
    df.to_csv('metafeatures_original_perf_new.csv')


if __name__ == '__main__':
    precalculate_performance()
    # print()
    # print(trials.best_trial['misc']['vals'])
        # trials = Trials()
        # as initial values pick points uniformly on the x-axis; we need this as a dict
        # init_vals = [config]
        # init_vals = [{'RF_PCA': 0.8, 'RF_preprocessing': 0,
        #               'classifier_type': 0, 'criterion': 0,
        #               'max_features': 0, 'min_samples_split': 0}]

        # this generates a trial object that can be used to bootstrap computation for the optimizer
        # trials = generate_trials_to_calculate(init_vals)
        # data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
        # fmin_objective = partial(objective, data=data)
        # best = fmin(fn=fmin_objective, space=space, algo=tpe.suggest, max_evals=3, trials=trials)
        # x_prime = space_eval(space, best)



