import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import openml
from pymfe.mfe import MFE
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
import time
import shutil


# Downloads and processes datasets used in [Feurer et al. 2015] from OpenML,
# extracts metafeatures according to the paper.#


pd.options.display.max_rows = 999
pd.options.display.max_columns = 50


paper_datasets_names = [
    'abalone',
    'anneal',
    'arrhythmia',
    'audiology',
    'autos',
    'balance-scale',
    'braziltourism',
    'breast-cancer',
    'breast-w',
    'car',
    'cmc',
    'credit-approval',
    'credit-g',
    'cylinder-bands',
    'dermatology',
    'diabetes',
    'ecoli',
    'eucalyptus',
    'glass',
    'haberman',
    'heart-c',
    'heart-h',
    'heart-statlog',
    'hepatitis',
    'ionosphere',
    'iris',
    'kr-vs-kp',
    'labor',
    'letter',
    'liver-disorders',
    'lymph',
    'mfeat-factors',
    'mfeat-fourier',
    'mfeat-karhunen',
    'mfeat-morphological',
    'mfeat-pixel',
    'mfeat-zernike',
    'mushroom',
    'nursery',
    'optdigits',
    'page-blocks',
    'pendigits',
    'postoperative-patient-data',
    'primary-tumor',
    'satimage',
    'segment',
    'sonar',
    'soybean',
    'spambase',
    'tae',
    'tic-tac-toe',
    'vehicle',
    'vote',
    'vowel',
    'waveform-5000',
    'yeast',
    'zoo'
]


openml_metafeatures_names = [
    'AutoCorrelation',
    'CfsSubsetEval_DecisionStumpAUC',
    'CfsSubsetEval_DecisionStumpErrRate',
    'CfsSubsetEval_DecisionStumpKappa',
    'CfsSubsetEval_NaiveBayesAUC',
    'CfsSubsetEval_NaiveBayesErrRate',
    'CfsSubsetEval_NaiveBayesKappa',
    'CfsSubsetEval_kNN1NAUC',
    'CfsSubsetEval_kNN1NErrRate',
    'CfsSubsetEval_kNN1NKappa',
    'ClassEntropy',
    'DecisionStumpAUC',
    'DecisionStumpErrRate',
    'DecisionStumpKappa',
    'Dimensionality',
    'EquivalentNumberOfAtts',
    'J48.00001.AUC',
    'J48.00001.ErrRate',
    'J48.00001.Kappa',
    'J48.0001.AUC',
    'J48.0001.ErrRate',
    'J48.0001.Kappa',
    'J48.001.AUC',
    'J48.001.ErrRate',
    'J48.001.Kappa',
    'MajorityClassPercentage',
    'MajorityClassSize',
    'MaxAttributeEntropy',
    'Max KurtosisOfNumericAtts',
    'MaxMeansOfNumericAtts',
    'MaxMutualInformation',
    'MaxNominalAttDistinctValues',
    'MaxSkewnessOfNumericAtts',
    'MaxStdDevOfNumericAtts',
    'MeanAttributeEntropy',
    'MeanKurtosisOfNumericAtts',
    'MeanMeansOfNumericAtts',
    'MeanMutualInformation',
    'MeanNoiseToSignalRatio',
    'MeanNominalAttDistinctValues',
    'MeanSkewnessOfNumericAtts',
    'MeanStdDevOfNumericAtts',
    'MinAttributeEntropy',
    'MinKurtosisOfNumericAtts',
    'MinMeansOfNumericAtts',
    'MinMutualInformation',
    'MinNominalAttDistinctValues',
    'MinSkewnessOfNumericAtts',
    'MinStdDevOfNumericAtts',
    'MinorityClassPercentage',
    'MinorityClassSize',
    'NaiveBayesAUC',
    'NaiveBayesErrRate',
    'NaiveBayesKappa',
    'NumberOfBinaryFeatures',
    'NumberOfClasses',
    'NumberOfFeatures',
    'NumberOfInstances',
    'NumberOfInstancesWithMissingValues',
    'NumberOfMissingValues',
    'NumberOfNumericFeatures',
    'NumberOfSymbolicFeatures',
    'PercentageOfBinaryFeatures',
    'PercentageOfInstancesWithMissingValues',
    'PercentageOfMissingValues',
    'PercentageOfNumericFeatures',
    'PercentageOfSymbolicFeatures',
    'Quartile1AttributeEntropy',
    'Quartile1KurtosisOfNumericAtts',
    'Quartile1MeansOfNumericAtts',
    'Quartile1MutualInformation',
    'Quartile1SkewnessOfNumericAtts',
    'Quartile1StdDevOfNumericAtts',
    'Quartile2AttributeEntropy',
    'Quartile2KurtosisOfNumericAtts',
    'Quartile2MeansOfNumericAtts',
    'Quartile2MutualInformation',
    'Quartile2SkewnessOfNumericAtts',
    'Quartile2StdDevOfNumericAtts',
    'Quartile3AttributeEntropy',
    'Quartile3KurtosisOfNumericAtts',
    'Quartile3MeansOfNumericAtts',
    'Quartile3MutualInformation',
    'Quartile3SkewnessOfNumericAtts',
    'Quartile3StdDevOfNumericAtts',
    'REPTreeDepth1AUC',
    'REPTreeDepth1ErrRate',
    'REPTreeDepth1Kappa',
    'REPTreeDepth2AUC',
    'REPTreeDepth2ErrRate',
    'REPTreeDepth2Kappa',
    'REPTreeDepth3AUC',
    'REPTreeDepth3ErrRate',
    'REPTreeDepth3Kappa',
    'RandomTreeDepth1AUC',
    'RandomTreeDepth1ErrRate',
    'RandomTreeDepth1Kappa',
    'RandomTreeDepth2AUC',
    'RandomTreeDepth2ErrRate',
    'RandomTreeDepth2Kappa',
    'RandomTreeDepth3AUC',
    'RandomTreeDepth3ErrRate',
    'RandomTreeDepth3Kappa',
    'StdvNominalAttDistinctValues',
    'kNN1NAUC',
    'kNN1NErrRate',
    'kNN1NKappa'
]

# the names of the metafeatures as they are listed in table 1 in the supplementary material to the paper
paper_metafeatures_names = [
    'class.entropy',
    'class.probability.max',
    'class.probability.mean',
    'class.probability.min',
    'class.probability.std',
    'dataset.ratio',
    'inverse.dataset.ratio',
    'kurtosis.max',
    'kurtosis.mean',
    'kurtosis.min',
    'kurtosis.sd',
    'landmarking.1NN',
    'landmarking.decision.node.learner',
    'landmarking.decision.tree',
    'landmarking.lda',
    'landmarking.naive.bayes',
    'landmarking.random.node.learner',
    'log.dataset.ratio',
    'log.inverse.dataset.ratio',
    'log.number.of.features',
    'log.number.of.instances',
    'number.of.instances.with.missing.values',
    'number.of.categorical.features',
    'number.of.classes',
    'number.of.features',
    'number.of.features.with.missing.values',
    'number.of.instances',
    'number.of.missing.values',
    'number.of.numeric.features',
    'pca.95percent',
    'pca.kurtosis.first.pc',
    'pca.skewness.first.pc',
    'percentage.of.instances.with.missing.values',
    'percentage.of.features.with.missing.values',
    'percentage.of.missing.values',
    'ratio.categorical.to.numerical',
    'ratio.numerical.to.categorical',
    'skewness.max',
    'skewness.mean',
    'skewness.min',
    'skewness.sd',
    'symbols.max',
    'symbols.mean',
    'symbols.min',
    'symbols.sd',
    'symbols.sum'
]



def _calculate_statistical_metafeatures(X, y, cat):
    X_num = X[:, np.where([not x for x in cat])[0]]
    if X_num.shape[1] != 0:
        sk = skew(X_num, axis=0)
        kurt = kurtosis(X_num, axis=1)
        stats = pd.Series({
                'skewness.max': sk.max(),
                'skewness.mean': sk.mean(),
                'skewness.min': sk.min(),
                'skewness.sd': sk.std(),
                'kurtosis.max': kurt.max(),
                'kurtosis.mean': kurt.mean(),
                'kurtosis.min': kurt.min(),
                'kurtosis.sd': kurt.std()
            })
    else:
        stats = pd.Series({
            'skewness.max': 0,
            'skewness.mean': 0,
            'skewness.min': 0,
            'skewness.sd': 0,
            'kurtosis.max': 0,
            'kurtosis.mean': 0,
            'kurtosis.min': 0,
            'kurtosis.sd': 0
        })

    X_cat = X[:, cat]
    if X_cat.shape[1] != 0:
        cat_cnts = np.nanmax(X_cat, axis=0)
        stats['symbols.max'] = np.nanmax(cat_cnts)
        stats['symbols.mean'] = np.nanmean(cat_cnts)
        stats['symbols.min'] = np.nanmin(cat_cnts)
        stats['symbols.sd'] = np.nanstd(X_cat)
        stats['symbols.sum'] = np.nansum(cat_cnts)
    else:
        stats['symbols.max'] = 0
        stats['symbols.mean'] = 0
        stats['symbols.min'] = 0
        stats['symbols.sd'] = 0
        stats['symbols.sum'] = 0
    return stats



def _calculate_pca_metafeatures(dataset):
    """
    [Bardenet et al. 2013] define rho as d' / d, where
    d' is the number of principal components explaining 95% of variance
    and d is the number of attributes.
    In addition to rho, the skewness and the kurtosis of each dataset projected
    onto its first principal component is calculated.
    """

    pca = PCA(.95, svd_solver='full')
    pca.fit(dataset)
    d_prime = pca.n_components_
    rho = d_prime / dataset.shape[1]
    projection_first_PC = pca.transform(dataset)[:, 0]
    kurtosis_projection = kurtosis(projection_first_PC)
    skewness_projection = skew(projection_first_PC)
    return pd.Series({
        'pca.95percent': rho,
        'pca.kurtosis.first.pc': kurtosis_projection,
        'pca.skewness.first.pc': skewness_projection
    })


def _calculate_landmarking_metafeatures(X, y):
    mfe = MFE(features=['one_nn',
                        'best_node',
                        'linear_discr',
                        'naive_bayes',
                        'random_node'], num_cv_folds=2)
    mfe.fit(X, y)
    ft = mfe.extract()
    res = dict(zip(ft[0], ft[1]))
    d = {'landmarking.1NN': res['one_nn.mean'],
         'landmarking.decision.node.learner': res['best_node.mean'],
         'landmarking.lda': res['linear_discr.mean'],
         'landmarking.naive.bayes': res['naive_bayes.mean'],
         'landmarking.random.node.learner': res['random_node.mean']}
    return pd.Series(d)


def _calculate_class_probabilities(labels):
    _, cl = np.unique(labels, return_counts=True)
    cl_prob = cl / cl.sum()
    res = {
        'class.probability.max': cl_prob.max(),
        'class.probability.mean': cl_prob.mean(),
        'class.probability.min': cl_prob.min(),
        'class.probability.std': cl_prob.std()
    }
    return pd.Series(res)


def calculate_paper_metafeatures(dataset):
    target = dataset.default_target_attribute
    X, y, cat, attr = dataset.get_data(target=target, dataset_format='array')
    dataset_metafeatures = pd.Series()
    dataset_metafeatures = dataset_metafeatures.append(_calculate_statistical_metafeatures(X, y, cat))
    dataset_metafeatures['number.of.features.with.missing.values'] = np.isnan(X).any(axis=0).sum()
    dataset_metafeatures['percentage.of.features.with.missing.values'] = dataset_metafeatures[
                                                                             'number.of.features.with.missing.values'] / \
                                                                         X.shape[1]
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    X = imputer.fit_transform(X)
    dataset_metafeatures = dataset_metafeatures.append(_calculate_pca_metafeatures(X))
    dataset_metafeatures = dataset_metafeatures.append(_calculate_landmarking_metafeatures(X, y))
    dataset_metafeatures = dataset_metafeatures.append(_calculate_class_probabilities(y))
    dataset_metafeatures['class.entropy'] = dataset.qualities['ClassEntropy']
    dataset_metafeatures['landmarking.decision.tree'] = dataset.qualities['DecisionStumpAUC']
    dataset_metafeatures['number.of.instances.with.missing.values'] = dataset.qualities[
        'NumberOfInstancesWithMissingValues']
    dataset_metafeatures['percentage.of.instances.with.missing.values'] = dataset.qualities[
        'PercentageOfInstancesWithMissingValues']
    dataset_metafeatures['percentage.of.missing.values'] = dataset.qualities['PercentageOfMissingValues']
    dataset_metafeatures['number.of.features'] = dataset.qualities['NumberOfFeatures']
    dataset_metafeatures['number.of.instances'] = dataset.qualities['NumberOfInstances']
    dataset_metafeatures['log.number.of.features'] = np.log(dataset_metafeatures['number.of.features'])
    dataset_metafeatures['log.number.of.instances'] = np.log(dataset_metafeatures['number.of.instances'])
    dataset_metafeatures['dataset.ratio'] = dataset_metafeatures['number.of.features'] / dataset_metafeatures[
        'number.of.instances']
    dataset_metafeatures['log.dataset.ratio'] = np.log(dataset_metafeatures['dataset.ratio'])
    dataset_metafeatures['inverse.dataset.ratio'] = 1.0 / dataset_metafeatures['dataset.ratio']
    dataset_metafeatures['log.inverse.dataset.ratio'] = np.log(dataset_metafeatures['inverse.dataset.ratio'])
    dataset_metafeatures['number.of.missing.values'] = dataset.qualities['NumberOfMissingValues']
    dataset_metafeatures['number.of.classes'] = dataset.qualities['NumberOfClasses']
    dataset_metafeatures['number.of.categorical.features'] = np.sum(np.array(cat))
    dataset_metafeatures['number.of.numeric.features'] = len(cat) - np.sum(np.array(cat))
    if dataset_metafeatures['number.of.numeric.features'] != 0:
        dataset_metafeatures['ratio.categorical.to.numerical'] = dataset_metafeatures[
                                                                     'number.of.categorical.features'] / \
                                                                 dataset_metafeatures['number.of.numeric.features']
    else:
        dataset_metafeatures['ratio.categorical.to.numerical'] = 1

    if dataset_metafeatures['ratio.categorical.to.numerical'] != 0:
        dataset_metafeatures['ratio.numerical.to.categorical'] = 1 / dataset_metafeatures[
            'ratio.categorical.to.numerical']
    else:
        dataset_metafeatures['ratio.numerical.to.categorical'] = 1
    return dataset_metafeatures



def download_paper_datasets():
    """
    Downloads datasets from Feurer et al. 2015, from OpenML.
    Impute nans with zeros and scale numerical features.

    :return:
    """

    print("Getting the list of all OpenML datasets.")
    openml_dfs = pd.DataFrame(openml.datasets.list_datasets()).transpose()
    print("Done.")
    if os.path.exists('data/datasets'):
        shutil.rmtree('data/datasets')
    os.makedirs('data/datasets')
    openml_dfs = openml_dfs[(openml_dfs['name'].isin(paper_datasets_names)) & (openml_dfs['version'] == 1)]
    openml_metafeatures_df = pd.DataFrame(columns=openml_metafeatures_names)
    paper_metafeatures_df = pd.DataFrame()

    for idx, row in openml_dfs.iterrows():
        start = time.time()
        print('Downloading {} ({})'.format(row['name'], idx))
        dataset = openml.datasets.get_dataset(str(row['did']), download_data=True, cache_format='pickle')
        target = dataset.default_target_attribute
        X, y, cat, attr = dataset.get_data(target=target, dataset_format='array')
        if X.shape[0] > 40000:
            print('Dataset is too large, skipping.')
            continue
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        X = imputer.fit_transform(X)
        if X[:, cat].shape[1] > 0:
            X_one_hot = OneHotEncoder(sparse=False).fit_transform(X[:, cat])
            X = np.delete(X, cat, axis=1)
            X = np.hstack([X, X_one_hot])
        if X.shape[1] > 0:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        os.makedirs('data/datasets/' + row['name'], exist_ok=True)
        pickle.dump(X_train, open('data/datasets/' + row['name'] + '/X_train.p', 'wb'))
        pickle.dump(y_train, open('data/datasets/' + row['name'] + '/y_train.p', 'wb'))
        pickle.dump(X_test, open('data/datasets/' + row['name'] + '/X_test.p', 'wb'))
        pickle.dump(y_test, open('data/datasets/' + row['name'] + '/y_test.p', 'wb'))

        # get new metafeatures from OpenML

        dataset_openml_mf = pd.Series(data=dataset.qualities)
        dataset_openml_mf['name'] = row['name']
        openml_metafeatures_df = openml_metafeatures_df.append(dataset_openml_mf, ignore_index=True)

        # calculate paper metafeatures

        dataset_paper_mf = calculate_paper_metafeatures(dataset)
        dataset_paper_mf['name'] = row['name']
        paper_metafeatures_df = paper_metafeatures_df.append(dataset_paper_mf, ignore_index=True)
        print('Done in {} seconds.'.format(time.time() - start))
    pickle.dump(paper_metafeatures_df, open('data/paper_metafeatures.p', 'wb'))
    pickle.dump(openml_metafeatures_df, open('data/openml_metafeatures.p', 'wb'))
    print('Saved paper metafeatures to data/paper_metafeatures.p')
    print('Saved OpenML metafeatures to data/openml_metafeatures.p')
    print('Saved preprocessed datasets to data/datasets/')

