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
import colorsys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from PIL import ImageColor

# CLASSIFIERS
from classifiers.dnn_classifier import DeepNeuralClassifier
from classifiers.dt_classifier import DecisionTreeClassifier
from classifiers.linsvc import SupportVectorMachineLinearKernelOneVsRestClassifier
from classifiers.nm_classifier import NearestMeanClassifier
from classifiers.knn_classifier import KNearestNeighborsClassifier
from classifiers.nvb_classifier import NaiveBayesClassifier
from classifiers.rf_classifier import RForestClassfier
from classifiers.svmlin_classifier import SupportVectorMachineLinearKernelClassifier
from classifiers.svmpol_classifier import SupportVectorMachinePolynomialKernelClassifier
from classifiers.svmrbf_classifier import SupportVectorMachineRbfKernelClassifier
from feature_selectors.NVBRFE_selector import RFESelector
from feature_selectors.all_selector import AllSelector
from feature_selectors.kruskall_selector import KruskallSelector
from feature_selectors.mann_whitney import MannWhitneySelector
from feature_selectors.perm_cor_selector import PermCorSelector
from feature_selectors.rand_selector import RandomSelector
from feature_selectors.gene_based_selector import GeneIndexSelector

# CONSTANTS

START_TIME = time.time()
BEANS_CONSTANT = 66
N_SAMPLES = 100  # number of samples (patients)
OUTER_FOLD = 10  # OUTER_FOLD-fold CV (outer loop) for triple-CV (Wessels, 2005: 3-fold)

classifiers = {
    'dnn': DeepNeuralClassifier,
    'nm': NearestMeanClassifier,
    'knn': KNearestNeighborsClassifier,
    'nvb': NaiveBayesClassifier,
    'dt': DecisionTreeClassifier,
    'rf': RForestClassfier,
    'svm_lin_or': SupportVectorMachineLinearKernelOneVsRestClassifier,
    'svm_lin_oo': SupportVectorMachineLinearKernelClassifier,
    'svm_pol': SupportVectorMachinePolynomialKernelClassifier,
    'svm_rbf': SupportVectorMachineRbfKernelClassifier
}

selectors = {
    'all': AllSelector,
    'rand': RandomSelector,
    'kruskall': KruskallSelector,
    'mannwhitney': MannWhitneySelector,
    'geneindex': GeneIndexSelector,
    'rfe': RFESelector
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
                train_acc = classifier.train(inner_train['features'][:, selected_indices], inner_train['labels'])
                accuracy = classifier.predict(inner_val['features'][:, selected_indices], inner_val['labels'])
                result = {'train_accuracy': train_acc, 'accuracy': accuracy, 'model': list(classifiers.values())[inner_i],
                          'selector': selector.__class__, 'indices': selected_indices}
                inner_accuracies.append(result)
                print(result)

                if accuracy > inner_best['accuracy']:
                    inner_best = result

                del classifier

            # Calculate and save accuracy of best classifier for current feature selector
            classifier = inner_best['model'](len(selected_indices), num_labels)
            print('[middle] Training %s / %s' % (classifier.__class__.__name__, selector.__class__.__name__))
            train_acc = classifier.train(middle_train['features'][:, selected_indices], middle_train['labels'])
            accuracy = classifier.predict(middle_val['features'][:, selected_indices], middle_val['labels'])

            if accuracy > middle_best['accuracy']:
                middle_best = {'train_accuracy': train_acc, 'accuracy': accuracy, 'model': inner_best['model'],
                               'selector': inner_best['selector'], 'indices': selected_indices}

            del classifier, selector

        # Calculate and save accuracy of best feature selector / classifier pair
        selector = middle_best['selector']()
        selected_indices = selector.select_features(outer_train['features'], outer_train['labels'])
        classifier = middle_best['model'](len(selected_indices), num_labels)
        print('[outer] Training %s / %s' % (classifier.__class__.__name__, selector.__class__.__name__))
        train_acc = classifier.train(outer_train['features'][:, selected_indices], outer_train['labels'])
        accuracy = classifier.predict(outer_val['features'][:, selected_indices], outer_val['labels'])
        result = {'train_accuracy': train_acc, 'accuracy': accuracy, 'model': middle_best['model'],
                  'selector': middle_best['selector'], 'indices': selected_indices}
        outer_accuracies.append(result)

        if accuracy > outer_best['accuracy']:
            outer_best = result

        del classifier, selector

    return outer_best, outer_accuracies, inner_accuracies

def make_faded(colorcode):
    """
    Takes a hex RGB color code, and returns a faded (less saturation) version of it.
    :param colorcode: Hex RGB string (e.g. #AABBCC)
    :return: Hex RGB string (e.g. #AABBCC)
    """
    r, g, b = ImageColor.getrgb(colorcode)
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    l = min([l * 1.5, 1.0])
    s *= 0.4
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return '#%02X%02X%02X' % (int(r*255), int(g*255), int(b*255))


def plot_accuracies(accuracies: list, title='Accuracies', hist_title='Selected features'):
    plt.figure()

    selected_indices = []

    field_names = ['accuracy', 'train_accuracy']
    for index, field_name in enumerate(field_names):
        grouped = {}
        for entry in accuracies:
            model, selector, accuracy, indices = entry['model'], entry['selector'], entry[field_name], entry['indices']
            key = '%s / %s' % (selector.__name__.replace('Selector', ''), model.__name__.replace('Classifier', ''))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(accuracy)
            selected_indices += indices

        means = [np.mean(group) for group in grouped.values()]
        stds = [np.std(group) for group in grouped.values()]
        names = ['%s / %d' % (key, len(grouped[key])) for key in grouped.keys()]
        available_colours = ['#0082c8', '#3cb44b', '#ffe119', '#f58231', '#e6194b', '#911eb4', '#d2f53c',
                             '#fabebe', '#008080', '#aa6e28', '#800000', '#ffd8b1']
        selectors = [name.split(' / ')[0] for name in names]
        unique_selectors = list(set(selectors))
        colours = [available_colours[unique_selectors.index(selector) % len(available_colours)] for selector in selectors]
        if index > 0:
            colours = [make_faded(colour) for colour in colours]
        ticks = [num * len(field_names) + index for num in range(0, len(means))]

        plt.bar(ticks, means, yerr=stds, color=colours)
        plt.xticks(ticks, names, rotation=90, fontsize=6)

    plt.xlabel('Selector/classifier pairs')
    plt.ylabel('Validation accuracy')
    plt.title(title)
    plt.subplots_adjust(bottom=0.4)
    plt.show()

    plt.hist(selected_indices)
    plt.xlabel('Selected feature index')
    plt.ylabel('Count')
    plt.title(hist_title)


def main():
    print('Script execution was initiated.')

    # Setting the seed (for reproducibility of training results)
    np.random.seed(0)

    if len(sys.argv) < 2 or sys.argv[1] != 'reuse':
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
        np.save('cache/best.npy', best)
        np.save('cache/outer_acc.npy', outer_acc)
        np.save('cache/inner_acc.npy', inner_acc)

        # TODO: Save model as *.pkl USING sklearn.joblib() ?
        # Train one last time on entire dataset
        model = create_final_model(best['model'], best['selector'], features, labels, num_unique_labels)
    else:
        best, outer_acc, inner_acc = np.load('cache/best.npy').item(), \
                                     np.load('cache/outer_acc.npy'), np.load('cache/inner_acc.npy')

    plot_accuracies(inner_acc, 'Inner fold accuracies')
    plot_accuracies(outer_acc, 'Outer fold accuracies')

    print('Triple-CV was finished.')
    print('Best performing pair (%f%%): %s / %s' % (best['accuracy'], best['selector'], best['model']))


# EXECUTION
if __name__ == '__main__':
    main()
    print('\nFinished: The script was successfully executed in %.8s s.' % (time.time() - START_TIME))
