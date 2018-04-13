#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kickstarted on Tuesday, 3rd of April, 2018.

This main script provides the backbone for the machine learning algorithm(s) and sets the framework for the validation-scheme.
Recommended Python interpreter is of version >=3.5.
Three-state classification into HER2+, TN or HR+.
Evaluation metric should be accuracy.
"""

# IMPORT OF LIBRARIES
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import pandas as pd
import math
import sys
from termcolor import colored

# CLASSIFIERS
from classifiers.dnn_classifier import DeepNeuralClassifier
from classifiers.nm_classifier import NearestMeanClassifier
from classifiers.knn_classifier import KNearestNeighborsClassifier
from classifiers.nvb_classifier import NaiveBayesClassifier
from feature_selectors.all_selector import AllSelector
from feature_selectors.kruskall_selector import KruskallSelector
from feature_selectors.mann_whitney import MannWhitneySelector
from feature_selectors.rand_selector import RandomSelector
from feature_selectors.gene_based_selector import GeneIndexSelector

# CONSTANTS

START_TIME = time.time()
BEANS_CONSTANT = 66
N_SAMPLES = 100  # number of samples (patients)
N_VARIABLES = 2834  # number of chromosomal locations
OUTER_FOLD = 4  # OUTER_FOLD-fold CV (outer loop) for triple-CV (Wessels, 2005: 3-fold)
MIDDLE_FOLD = 5  # INNER_FOLD-fold CV (inner loop) for triple-CV (Wessels, 2005: 10-fold)
INNER_FOLD = 5  # INNER_FOLD-fold CV (inner loop) for triple-CV (Wessels, 2005: 10-fold)

classifiers = {
    'dnn': DeepNeuralClassifier,
    'nm': NearestMeanClassifier,
    'knn': KNearestNeighborsClassifier,
    'nvb': NaiveBayesClassifier
}

selectors = {
    'all': AllSelector,
    'rand': RandomSelector,
    'kruskall': KruskallSelector,
    'mannwhitney': MannWhitneySelector,
    'geneindex': GeneIndexSelector
}


def get_data():
    """
    Import of raw data as np.array using Pandas.

    import of train_clinical.txt, samples x 2 (sample name, label)
    Sample example (100 in total, sample name): 'Array.129'
    Subgroups (3 in total, label): 'HER2+', 'HR+' and 'Triple Neg'
    :return: train_clinical, train_call
    """
    #
    train_clinical = pd.read_csv('data/Train_clinical.txt', sep='\t').values
    print('Data set "train_clinical" was loaded (%i rows and %i cols).' % (
        train_clinical.shape[0], train_clinical.shape[1]))

    # import and transpose of train_call.txt, samples x variables
    train_call = np.transpose(
        pd.read_csv('data/Train_call.txt', sep='\t', usecols=range(4, 4 + N_SAMPLES)).values.astype('float32'))
    print('Data set "train_call" was loaded (%i rows and %i cols).' % (train_call.shape[0], train_call.shape[1]))

    return train_call, train_clinical


def create_final_model(model_constructor, selector_constructor, features, labels, num_labels):
    selected_indices = selector_constructor().select_features(features, labels)
    model = model_constructor(len(selected_indices), num_labels)
    train_accuracy = model.train(features[:, selected_indices], labels)

    print('Final train accuracy: %f.' % train_accuracy)

    return model


def slice_data(features: list, labels: list, folds: int, current_fold: int) -> object:
    val_begin = int(current_fold / folds)
    val_end = int(val_begin + len(features) / folds)
    train_indices = list(range(0, val_begin)) + list(range(val_end, len(features)))
    val_indices = list(range(val_begin, val_end))
    train = {'features': features[train_indices], 'labels': labels[train_indices]}
    val = {'features': features[val_indices], 'labels': labels[val_indices]}
    return train, val


def triple_cross_validate(features: list, labels: list, num_labels: int):
    outer_fold, middle_fold, inner_fold = OUTER_FOLD, len(selectors), len(classifiers)
    outer_accuracies, inner_accuracies = [], []
    outer_best = {'accuracy': 0, 'model': None, 'selector': None}

    # Outer fold, used for accuracy validation of best selector/classifier pairs
    for outer_i in range(0, outer_fold):
        outer_train, outer_val = slice_data(features, labels, outer_fold, outer_i)
        middle_best = {'accuracy': 0, 'model': None, 'selector': None}

        # Middle fold, used for selecting the optimal selector
        for middle_i in range(0, middle_fold):
            middle_train, middle_val = slice_data(outer_train['features'], outer_train['labels'], middle_fold, middle_i)
            selector = list(selectors.values())[middle_i]()
            selected_indices = selector.select_features(middle_train['features'], middle_train['labels'])
            inner_best = {'accuracy': 0, 'model': None}

            # Inner fold, used for selecting the optimal classifier
            for inner_i in range(0, inner_fold):
                inner_train, inner_val = slice_data(middle_train['features'], middle_train['labels'], inner_fold,
                                                    inner_i)
                classifier = list(classifiers.values())[inner_i](len(selected_indices), num_labels)
                print('[inner] Training %s / %s' % (classifier.__class__.__name__, selector.__class__.__name__))
                classifier.train(inner_train['features'][:, selected_indices], inner_train['labels'])
                accuracy = classifier.predict(inner_val['features'][:, selected_indices], inner_val['labels'])
                result = {'accuracy': accuracy, 'model': list(classifiers.values())[inner_i],
                          'selector': selector.__class__}
                inner_accuracies.append(result)

                if accuracy > inner_best['accuracy']:
                    inner_best = result

            # Calculate and save accuracy of best classifier for current feature selector
            classifier = inner_best['model'](len(selected_indices), num_labels)
            print('[middle] Training %s / %s' % (classifier.__class__.__name__, selector.__class__.__name__))
            classifier.train(middle_train['features'][:, selected_indices], middle_train['labels'])
            accuracy = classifier.predict(middle_val['features'][:, selected_indices], middle_val['labels'])

            if accuracy > middle_best['accuracy']:
                middle_best = {'accuracy': accuracy, 'model': inner_best['model'], 'selector': inner_best['selector']}

        # Calculate and save accuracy of best feature selector / classifier pair
        selected_indices = middle_best['selector']().select_features(outer_train['features'], outer_train['labels'])
        classifier = middle_best['model'](len(selected_indices), num_labels)
        print('[outer] Training %s / %s' % (classifier.__class__.__name__, selector.__class__.__name__))
        classifier.train(outer_train['features'][:, selected_indices], outer_train['labels'])
        accuracy = classifier.predict(outer_val['features'][:, selected_indices], outer_val['labels'])
        result = {'accuracy': accuracy, 'model': middle_best['model'], 'selector': middle_best['selector']}
        outer_accuracies.append(result)

        if accuracy > outer_best['accuracy']:
            outer_best = result

    return outer_best, outer_accuracies, inner_accuracies


def plot_accuracies(accuracies: list, title='Accuracies'):
    grouped = {}

    for entry in accuracies:
        model, selector, accuracy = entry['model'], entry['selector'], entry['accuracy']
        key = '%s / %s' % (selector.__name__.replace('Selector', ''), model.__name__.replace('Classifier', ''))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(accuracy)

    means = [np.mean(group) for group in grouped.values()]
    stds = [np.std(group) for group in grouped.values()]
    names = ['%s / %d' % (key, len(grouped[key])) for key in grouped.keys()]
    ticks = range(0, len(means))

    plt.figure()
    plt.bar(ticks, means, yerr=stds)
    plt.xlabel('Selector/classifier pairs')
    plt.ylabel('Validation accuracy')
    plt.title(title)
    plt.xticks(ticks, names, rotation=20, fontsize=6)
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    # for entry in grouped:


def main():
    print('Script execution was initiated.')

    # Setting the seed (for reproducibility of training results)
    np.random.seed(0)

    # The order in both np.arrays is the same as in the original files, which means that the label (output) \\
    # train_clinical[a, 1] is the wanted prediction for the data (features) in train_call[a, :]"""
    train_call, train_clinical = get_data()
    features, labels = train_call, train_clinical[:, 1]

    if len(features) != len(labels):
        sys.exit('Data and response files do not have the same amount of lines')

    # TODO: Data pre-processing

    # test if provided constants INNER_FOLD and OUTER_FOLD are allowed
    if not (N_SAMPLES % OUTER_FOLD == 0):
        print('OUTER_FOLD constant is not appropriate.')
        print('Script execution is aborted after %.8s s.' % (time.time() - START_TIME))
        sys.exit()

    # Select model to run, based on command line parameter
    feature_length, num_unique_labels = features.shape[1], len(set(labels))

    # give standard output
    print('Triple cross-validation with %i-fold and subsequent %i-fold and %i-fold splits is initiated.' %
          (OUTER_FOLD, len(selectors.keys()), len(classifiers.keys())))

    best, outer_acc, inner_acc = triple_cross_validate(features, labels, num_unique_labels)
    plot_accuracies(inner_acc, 'Inner fold accuracies')
    plot_accuracies(outer_acc, 'Outer fold accuracies')

    print('Triple-CV was finished.')
    print('Best performing pair (%f%%): %s / %s' % (best['accuracy'], best['selector'], best['model']))


    # Train one last time on entire dataset
    model = create_final_model(best['model'], best['selector'], features, labels, num_unique_labels)

    # TODO: Save model as *.pkl USING sklearn.joblib() ?


# EXECUTION
if __name__ == '__main__':
    main()
    print('\nFinished: The script was successfully executed in %.8s s.' % (time.time() - START_TIME))
