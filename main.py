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
# from classifiers.nm_classifier import NearestMeanClassifier
# from classifiers.knn_classifier import KNearestNeighborsClassifier
from classifiers.naivebayses import NaivesBayes
from feature_selectors.all_selector import AllSelector
from feature_selectors.rand_selector import RandomSelector


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
    # 'nm': NearestMeanClassifier,
    # 'knn': KNearestNeighborsClassifier,
    'nvb': NaivesBayes
}

selectors = {
    'all': AllSelector,
    'rand': RandomSelector
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


def plot_accuracy(model, train_accuracy, train_accuracy_mean, val_accuracy, val_accuracy_mean):
    """
    Plots the accuracies of all rounds of a triple cross validation
    :param train_accuracy:
    :param val_accuracy:
    :return:
    """
    print('Training accuracy: %.4f.' % train_accuracy_mean)
    print(colored('Validation accuracy: %.4f.', 'green') % val_accuracy_mean + " You 'mirin, bra?")
    print('The distribution of the evaluation metric (accuracy) is being plotted.')

    plt.figure()
    plt.plot(train_accuracy, alpha=0.4, label='training accuracies')
    plt.plot(val_accuracy, alpha=0.4, label='validation accuracies')
    plt.xlabel('Cross-validation round')
    plt.ylabel('Accuracy')
    plt.title('Distribution of accuracies in three-fold CV for %s' % model.__class__.__name__)
    plt.legend()
    plt.show()


def create_final_model(model_constructor, features, labels, selected_indices, num_labels):
    model = model_constructor(len(selected_indices), num_labels)
    train_accuracy = model.train(features[:, selected_indices], labels)

    print('Final train accuracy: %f.' % train_accuracy)

    return model

def slice_data(features, labels, folds, current_fold):
    train_indices = range(0, current_fold * len(features)) + range((current_fold + 1) * len(features), len(features))
    val_indices = range(current_fold * len(features), (current_fold + 1) * len(features))
    train = {'features': features[train_indices], 'labels': labels[train_indices]}
    val = {'features': features[val_indices], 'labels': labels[val_indices]}
    return train, val

def triple_cross_validate(features, labels, num_labels):
    outer_fold, middle_fold, inner_fold = OUTER_FOLD, len(selectors), len(classifiers)

    for outer_i in range(0, outer_fold):
        outer_train, outer_val = slice_data(features, labels, outer_fold, outer_i)

        for middle_i in range(0, middle_fold):
            middle_train, middle_val = slice_data(outer_train['features'], outer_train['labels'], middle_fold, middle_i)
            selector = selectors[middle_i]()
            selected_indices = selector.select_features(middle_train['features'])

            for inner_i in range(0, inner_fold):
                inner_train, inner_val = slice_data(middle_train['features'], middle_train['labels'], inner_fold, inner_i)
                classifier = classifiers[inner_i](len(selected_indices), num_labels)
                accuracy = classifier.train(inner_train['features'][:, selected_indices], inner_train['labels'])

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

    # Triple cross-validation (with random sampling without replacement) (similar to Wessels, 2005)
    # Hyper-parameter selection can be integrated (e. g. k in kNN)
    # (ALTERNATIVE: WITH REPLACEMENT, then other OUTER_FOLD AND INNER_FOLD are allowed)
    # test if provided constants INNER_FOLD and OUTER_FOLD are allowed
    if not (N_SAMPLES % OUTER_FOLD == 0 and N_SAMPLES / OUTER_FOLD % INNER_FOLD == 0):
        print('INNER_FOLD and OUTER_FOLD constants are not appropriate.')
        print('Script execution is aborted after %.8s s.' % (time.time() - START_TIME))
        sys.exit()

    if len(sys.argv) != 3 or sys.argv[1] not in selectors.keys() or sys.argv[2] not in classifiers.keys():
        sys.exit('Usage: python main.py [%s] [%s]' % ('|'.join(selectors.keys()), '|'.join(classifiers.keys())))

    # Select model to run, based on command line parameter
    feature_length, num_unique_labels = features.shape[1], len(set(labels))
    selector = selectors[sys.argv[1]]()
    model_constructor = classifiers[sys.argv[2]]

    # give standard output
    print('Triple cross-validation with %i-fold and subsequent %i-fold split is initiated.' % (OUTER_FOLD, INNER_FOLD))

    train_accuracy, train_accuracy_mean, val_accuracy, val_accuracy_mean, \
        selected_indices = cross_validate(selector, model_constructor, features, labels, num_unique_labels)

    print('Triple-CV was finished.')
    plot_accuracy(model_constructor, train_accuracy, train_accuracy_mean, val_accuracy, val_accuracy_mean)

    # Show model accuracy on entire dataset
    print('Showing accuracy of model on entire dataset')
    final_train_acc, final_train_mean, final_val_acc, final_val_mean = \
        calc_final_accuracy(features[:, selected_indices], labels, model_constructor(len(selected_indices), num_unique_labels))
    plot_accuracy(model_constructor, final_train_acc, final_train_mean, final_val_acc, final_val_mean)

    # Train one last time on entire dataset
    model = create_final_model(model_constructor, features, labels, selected_indices, num_unique_labels)

    # TODO: Save model as *.pkl USING sklearn.joblib() ?


# EXECUTION
if __name__ == '__main__':
    main()
    print('\nFinished: The script was successfully executed in %.8s s.' % (time.time() - START_TIME))
