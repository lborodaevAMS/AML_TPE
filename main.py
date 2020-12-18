from hyperopt import hp, fmin, tpe
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


space = hp.choice('classifier_type', [
    {
        'type': 'naive_bayes',
    },
    {
        'type': 'svm',
        'C': hp.lognormal('SVM_C', 0, 1.0),
        'ktype': hp.choice('svm_kernel', [
            {'kernel': 'linear'},
            {'kernel': 'rbf', 'gamma': hp.lognormal('svm_rbf_width', 0, 1)},
        ]),
    },
])

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def objective(args):
    classifier_type = args['type']
    del args['type']
    if classifier_type == 'naive_bayes':
        clf = MultinomialNB()
    elif classifier_type == 'svm':
        ktype = args['ktype']
        del args['ktype']
        args = dict(args, **ktype)
        print(args)
        clf = SVC(**args)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return (y_pred == y_test).mean()


if __name__ == '__main__':
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

