from hyperopt import hp, fmin, tpe, Trials, space_eval, rand
from hyperopt.fmin import generate_trials_to_calculate
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial
import typing
import seaborn as sns
from PrepareData import download_paper_datasets
import os



sns.set()

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


def __unpack_item_vals(items_vals):
    conf = dict()
    for k, v in items_vals.items():
        if len(v) > 0:
            conf[k] = v[0]
    return conf


def __instantiate_classifier(args):
    preprocess = args['preprocessing']
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
        pca = PCA(n_components=preprocess, svd_solver='full')
        pipe = Pipeline([('pca', pca), ('clf', clf)])
    else:
        pipe = Pipeline([('clf', clf)])
    return pipe


@ignore_warnings(category=ConvergenceWarning)
def objective(args, data):
    X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'])
    clf = __instantiate_classifier(args)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    err = (y_pred != y_test).mean()
    del clf
    return err



def _load_dataset(path_prefix: str) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = pickle.load(open(path_prefix + '/X_train.p', 'rb'))
    X_test = pickle.load(open(path_prefix + '/X_test.p', 'rb'))
    y_train = pickle.load(open(path_prefix + '/y_train.p', 'rb'))
    y_test = pickle.load(open(path_prefix + '/y_test.p', 'rb'))
    assert X_train.shape[1] == X_test.shape[1]
    return X_train, X_test, y_train, y_test



def random_search():
    datasets = os.listdir('data/datasets')
    df = pd.DataFrame()
    print('Finding the best configs for {} datasets using Random Search...'.format(len(datasets)))
    for i, d_name in enumerate(os.listdir('data/datasets')):
        print('Processing {} ({} out of {})...'.format(d_name, i + 1, len(datasets)))
        X_train, X_test, y_train, y_test = _load_dataset('data/datasets/' + d_name)
        trials = Trials()
        data = {'X': X_train, 'y': y_train}
        fmin_objective = partial(objective, data=data)
        fmin(fn=fmin_objective, space=space, algo=rand.suggest, max_evals=40, trials=trials)
        s = pd.Series(index=['RF_PCA', 'RF_preprocessing', 'SVM_C', 'SVM_Gamma', 'SVM_Linear_C',
                             'SVM_Linear_PCA', 'SVM_Linear_Penalty', 'SVM_Linear_preprocessing',
                             'SVM_RBF_PCA', 'SVM_RBF_preprocessing', 'classifier_type', 'criterion',
                             'max_features', 'min_samples_split'])
        # for k, v in trials.best_trial['misc']['idxs'].items():
        #     if len(v) > 0:
        #         s[k] = v[0]
        for k, v in trials.best_trial['misc']['vals'].items():
            if len(v) > 0:
                s[k] = v[0]
        s['name'] = d_name
        df = df.append(s, ignore_index=True)
    pickle.dump(df, open('data/warm_start_data.p', 'wb'))


def _read_config(row: pd.Series) -> dict:
    config = dict()
    for k in [
        'RF_PCA', 'RF_preprocessing', 'SVM_C', 'SVM_Gamma', 'SVM_Linear_C', 'SVM_Linear_PCA', 'SVM_Linear_Penalty',
        'SVM_Linear_preprocessing', 'SVM_RBF_PCA', 'SVM_RBF_preprocessing', 'classifier_type', 'criterion',
        'max_features', 'min_samples_split'
    ]:
        if not np.isnan(row[k]):
            config[k] = row[k]
    return config


def plot_validation_performance(dataset_name: str):

    train_data = {
        'X': pickle.load(open('original_datasets/' + dataset_name + '/X_train.p', 'rb')),
        'y': pickle.load(open('original_datasets/' + dataset_name + '/y_train.p', 'rb')),
    }
    val_data = {
        'X': pickle.load(open('original_datasets/' + dataset_name + '/X_test.p', 'rb')),
        'y': pickle.load(open('original_datasets/' + dataset_name + '/y_test.p', 'rb')),
    }

    train_objective = partial(objective, data=train_data)
    val_objective = partial(objective, data=val_data)

    # Evaluate without warm-starting
    val_loss = np.zeros(shape=(10, 20))
    # 5-fold CV
    for i in range(5):
        trials = Trials()
        fmin(fn=train_objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
        for j in range(20):
            conf = __unpack_item_vals(trials.trials[0]["misc"]["vals"])
            magic_str = space_eval(space, conf)
            val_loss[i, j] = val_objective(magic_str)

    # Evaluate with warm-starring
    init_vals = find_init_vals(dataset_name)
    # 5-fold CV
    for i in range(5, 10):
        trials = generate_trials_to_calculate(init_vals)
        fmin(fn=train_objective, space=space, algo=tpe.suggest, max_evals=15, trials=trials)
        for j in range(20):
            conf = __unpack_item_vals(trials.trials[j]["misc"]["vals"])
            magic_str = space_eval(space, conf)
            val_loss[i, j] = val_objective(magic_str)

    y = np.minimum.accumulate(val_loss, axis=1)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(0, 20, 5))
    ax.plot(range(y.shape[1]), y[0:5, :].mean(axis=0), markersize=5, c='darkgoldenrod', marker="s", label='TPE')
    ax.fill_between(range(y.shape[1]), y[0:5, :].mean(axis=0) - y[0:5, :].std(axis=0), y[0:5, :].mean(axis=0) +
                     y[0:5, :].std(axis=0), alpha=0.3, color='darkgoldenrod')

    ax.plot(range(y.shape[1]), y[5:10, :].mean(axis=0), markersize=5, c='blue', marker="s")
    ax.fill_between(range(y.shape[1]), y[5:10, :].mean(axis=0) - y[5:10, :].std(axis=0),
                     y[5:10, :].mean(axis=0) + y[5:10, :].std(axis=0), alpha=0.3,
                     color='blue', label='MI-TPE')
    ax.set_xlabel('# function evaluations')
    ax.set_ylabel('validation error')
    ax.legend()
    fig.savefig(dataset_name + '.png', dpi=300)
    plt.show()


def find_init_vals(dataset_name, use_old_metafeatures=False):
    params = pickle.load(open('data/warm_start_data.p', 'rb'))
    config_dim = params.shape[1]
    if use_old_metafeatures:
        metafeatures = pickle.load(open('data/paper_metafeatures.p', 'rb'))
    else:
        metafeatures = pickle.load(open('data/openml_metafeatures.p', 'rb'))

    # align all data in the same order
    m = pd.merge(params, metafeatures, on='name')
    query = m[m['name'] == dataset_name].iloc[0, config_dim:].fillna(0).values
    params = m[m['name'] != dataset_name].iloc[:, 0:config_dim - 1]
    metafeatures = m[m['name'] != dataset_name].iloc[:, config_dim:].fillna(0).values
    kdtree = KDTree(metafeatures)
    d, idx = kdtree.query(query, 5)
    init_vals = [_read_config(params.iloc[i, :]) for i in idx]
    return init_vals


def __eval_tpe(dataset_name: str, k, warm_start=False, use_old_metafeatures=False) -> np.ndarray:
    train_data = {
        'X': pickle.load(open('data/datasets/' + dataset_name + '/X_train.p', 'rb')),
        'y': pickle.load(open('data/datasets/' + dataset_name + '/y_train.p', 'rb')),
    }
    val_data = {
        'X': pickle.load(open('data/datasets/' + dataset_name + '/X_test.p', 'rb')),
        'y': pickle.load(open('data/datasets/' + dataset_name + '/y_test.p', 'rb')),
    }
    train_objective = partial(objective, data=train_data)
    val_objective = partial(objective, data=val_data)

    val_loss = np.zeros(shape=(k, 20))
    # k-fold CV
    print('Doing 5-fold CV on {}...'.format(dataset_name))
    for i in range(k):
        print('Validating fold {}'.format(i))
        evals = 20
        trials = Trials()
        if warm_start:
            init_vals = find_init_vals(dataset_name, use_old_metafeatures)
            trials = generate_trials_to_calculate(init_vals)
            evals = 15
        fmin(fn=train_objective, space=space, algo=tpe.suggest, max_evals=evals, trials=trials, timeout=20)
        for j in range(20):
            conf = __unpack_item_vals(trials.trials[0]["misc"]["vals"])
            magic_str = space_eval(space, conf)
            val_loss[i, j] = val_objective(magic_str)
    return val_loss



def benchmark_mi_tpe(k, n=1.0):
    d_names = os.listdir('data/datasets/')
    assert n < len(d_names)
    d_names = np.random.choice(d_names, size=n, replace=False)
    val_err_tpe = np.full((len(d_names) * k, 20), np.inf)
    val_err_mi_tpe = np.full((len(d_names) * k, 20), np.inf)

    for i, name in enumerate(d_names):
        val_err_tpe[i * k: (i + 1) * k, :] = __eval_tpe(name, k, warm_start=False, use_old_metafeatures=True)
        val_err_mi_tpe[i * k: (i + 1) * k, :] = __eval_tpe(name, k, warm_start=True, use_old_metafeatures=True)


    pickle.dump(val_err_tpe, open('data/val_error_tpe.p', 'wb'))
    pickle.dump(val_err_mi_tpe, open('data/val_error_mi_tpe.p', 'wb'))


    y = np.minimum.accumulate(val_err_tpe, axis=1)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(0, 20, 5))
    ax.plot(range(y.shape[1]), y.mean(axis=0), markersize=5, c='darkgoldenrod', marker="s", label='TPE')
    ax.fill_between(range(y.shape[1]), y.mean(axis=0) - y.std(axis=0), y.mean(axis=0) +
                     y.std(axis=0), alpha=0.3, color='darkgoldenrod')

    y = np.minimum.accumulate(val_err_mi_tpe, axis=1)
    ax.plot(range(y.shape[1]), y.mean(axis=0), markersize=5, c='blue', marker="s")
    ax.fill_between(range(y.shape[1]), y.mean(axis=0) - y.std(axis=0),
                     y.mean(axis=0) + y.std(axis=0), alpha=0.3, color='blue', label='MI-TPE')
    ax.set_xlabel('# function evaluations')
    ax.set_ylabel('validation error')
    ax.legend()
    fig.savefig('img/tpe_vs_mitpe.png', dpi=300)
    plt.show()



def benchmark_metafeatures():
    d_names = os.listdir('data/datasets/')
    assert n < len(d_names)
    d_names = np.random.choice(d_names, size=n, replace=False)
    val_err_old = np.full((len(d_names) * k, 20), np.inf)
    val_err_new = np.full((len(d_names) * k, 20), np.inf)

    for i, name in enumerate(d_names):
        val_err_old[i * k: (i + 1) * k, :] = __eval_tpe(name, k, warm_start=True, use_old_metafeatures=True)
        val_err_new[i * k: (i + 1) * k, :] = __eval_tpe(name, k, warm_start=True, use_old_metafeatures=False)

    pickle.dump(val_err_old, open('data/val_error_old_mf.p', 'wb'))
    pickle.dump(val_err_new, open('data/val_error_new_mf.p', 'wb'))

    y = np.minimum.accumulate(val_err_old, axis=1)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(0, 20, 5))
    ax.plot(range(y.shape[1]), y.mean(axis=0), markersize=5, c='darkgoldenrod',
            marker="s", label='MI-TPE, old metafeatures')
    ax.fill_between(range(y.shape[1]), y.mean(axis=0) - y.std(axis=0), y.mean(axis=0) +
                    y.std(axis=0), alpha=0.3, color='darkgoldenrod')

    y = np.minimum.accumulate(val_err_new, axis=1)
    ax.plot(range(y.shape[1]), y.mean(axis=0), markersize=5, c='blue', marker="s")
    ax.fill_between(range(y.shape[1]), y.mean(axis=0) - y.std(axis=0),
                    y.mean(axis=0) + y.std(axis=0), alpha=0.3, color='blue', label='MI-TPE, new metafeatures')
    ax.set_xlabel('# function evaluations')
    ax.set_ylabel('validation error')
    ax.legend()
    fig.savefig('img/old_vs_new_metafeatures.png', dpi=300)
    plt.show()




if __name__ == '__main__':
    download_paper_datasets()
    random_search()

    # k = number of folds
    k = 5

    # n = number of datasets to consider
    # note that the data from all evaluated datasets it still used for warmstarting the problem
    n = 2

    benchmark_mi_tpe(k, n)
    benchmark_metafeatures(k, n)

