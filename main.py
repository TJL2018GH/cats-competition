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
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from PIL import ImageColor

# CLASSIFIERS AND FEATURE SELECTORS
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
from classifiers.gba_classifier import GradientBoostingAlgorithm
from classifiers.lr_classifier import LogisticRegressionClassifier
from ensemble.voting_ensemble import VotingEnsemble
from ensemble.best_ensemble import BestEnsemble
from feature_selectors.base_selector import BaseSelector
from feature_selectors.NVBRFE_selector import RFESelector
from feature_selectors.all_selector import AllSelector
from feature_selectors.kruskall_selector import KruskallSelector
from feature_selectors.mann_whitney import MannWhitneySelector
from feature_selectors.perm_cor_selector import PermCorSelector
from feature_selectors.rand_selector import RandomSelector
from feature_selectors.gene_based_selector import GeneIndexSelector

# IMPORT FOR CROSSVAL
from sklearn.model_selection import train_test_split
# CONSTANTS

START_TIME = time.time()
BEANS_CONSTANT = 66
N_SAMPLES = 100  # number of samples (patients)
OUTER_FOLD = 3  # OUTER_FOLD-fold CV (outer loop) for triple-CV (Wessels, 2005: 3-fold)
MIDDLE_FOLD = 6
INNER_FOLD = 5

# Comment any classifiers that will not be included
classifiers = {
    'dnn': DeepNeuralClassifier,
    'nm': NearestMeanClassifier,
    'knn': KNearestNeighborsClassifier,
    'nvb': NaiveBayesClassifier,
    'dt': DecisionTreeClassifier,
    'rf': RForestClassfier,
    'lg': LogisticRegressionClassifier,
    'svm_lin_or': SupportVectorMachineLinearKernelOneVsRestClassifier,
    'svm_lin_oo': SupportVectorMachineLinearKernelClassifier,
    'svm_pol': SupportVectorMachinePolynomialKernelClassifier,
    'svm_rbf': SupportVectorMachineRbfKernelClassifier,
    'gba_ens': GradientBoostingAlgorithm,
    'vot_ens': VotingEnsemble
}

# Comment any feature selectors that will not be included
selectors = {
    'all': AllSelector,
    'rand': RandomSelector,
    'kruskall': KruskallSelector,
    'mannwhitney': MannWhitneySelector,
    'geneindex': GeneIndexSelector,
    'rfe': RFESelector
}

# Ensemble methods to be used
ensembles = {
    'best_ens': BestEnsemble
}

# Fix pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

def labels_to_categorical(labels):
    _,IDs = np.unique(labels,return_inverse=True)
    return IDs


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


def stratification(labels, n_fold, n_class):

    labels = labels_to_categorical(labels)
    tot_num_samples = len(labels)
    sample_pool = pd.DataFrame({'labels': labels ,'position': range(0, tot_num_samples)})
    group_pool = sample_pool.groupby('labels')

    fold_size = round(float(tot_num_samples/(n_fold*n_class)),0)
    classes_pool = []
    folds_by_class = []
    smallest_class = len(group_pool.get_group(0))
    idx_smallest = 0

    # find the smallest class and create classes chunk:

    for class_idx in range(n_class):

        classes_pool.append(group_pool.get_group(class_idx))
        if smallest_class >= len(group_pool.get_group(class_idx)):
            smallest_class = len(group_pool.get_group(class_idx))
            idx_smallest = class_idx

    # create the balanced fold
    for class_chunk in classes_pool:

        balanced_class = class_chunk.sample(n=smallest_class)

        temp_chunks=[]
        for fold in range(n_fold):

            fold_chunk = balanced_class[int(fold):int(fold + fold_size) ]
            fold_chunk["fold"] = np.ones(int(fold_size)) * int(fold)

            temp_chunks.append(fold_chunk)
        folds_by_class.append(pd.concat(temp_chunks))

    cross_validation_folds = pd.concat(folds_by_class)
    cross_validation_folds_ordered = cross_validation_folds.sort_values('fold')

    return np.array(cross_validation_folds_ordered['position']),(tot_num_samples-len((cross_validation_folds_ordered['position'])))


def create_final_model(model_dict, selector_constructor, features, labels, num_labels, middle_accu):
    """
    Train the best performing model on the entire data set

    :param: model_dict: Dictionary with the different models
    :param: selector_constructor: Feature_selection method
    :param: features: Vector of features of the entire data set
    :param: labels: Labels of the entire dataset
    :param: num_labels: Number of unique labels
    :param: middle_accu: Accuracies of the middle fold
    :return: model: Model trained on the entire dataset
    """
    model_constructor = model_dict['model']
    selected_indices = selector_constructor().select_features(features, labels)
    model = model_constructor(len(selected_indices), num_labels)
    print(model_dict['model_name'])
    if model_dict['model_name'] == 'best_ens':
        model.add_combinations_list(middle_accu)
        selected_indices = model_dict['indices']

    train_accuracy = model.train(features[:, selected_indices], labels)

    print('Final train accuracy: %f.' % train_accuracy)

    return model


def get_best_performing(results):
    """
    Returns the best performing entry in a list of validation accuracies.
    :param results: List of validation accuracies
    :return: Best performing entry
    """
    if len(results) == 0:
        print("Warning: get_best_performance() called with empty results list")
        return None

    accuracies = {}

    for index, entry in enumerate(results):
        model_name = entry['model_name']
        selector_name = entry['selector'].__name__
        key = model_name + selector_name

        if not key in accuracies:
            accuracies[key] = {
                'model': entry['model'], 'model_name': model_name, 'selector': entry['selector'],
                'indices': entry['indices'], 'train_accuracy': [], 'accuracy': []
            }

        accuracies[key]['accuracy'].append(entry['accuracy']);
        accuracies[key]['train_accuracy'].append(entry['train_accuracy']);

    best = {'accuracy': 0}

    for entry in accuracies.values():
        entry['accuracy'] = np.mean(entry['accuracy'])
        entry['train_accuracy'] = np.mean(entry['train_accuracy'])

        if entry['accuracy'] > best['accuracy']:
            best = entry

    return best


def slice_data(features: list, labels: list, folds: int, current_fold: int) -> object:
    """
    Split data into training and validation set

    :param: features: Data to be split
    :param: labels: List of the labels corresponding to the rows of the features
    :param: folds: Number of folds that the data will be split into
    :param: current_fold: Current fold in the loop calling the function
    :return: train: Training data
    :return: val: Validation data
    """
    val_begin = int(current_fold / folds)
    val_end = int(val_begin + len(features) / folds)
    train_indices = list(range(0, val_begin)) + list(range(val_end, len(features)))
    val_indices = list(range(val_begin, val_end))
    train = {'features': features[train_indices], 'labels': labels[train_indices]}
    val = {'features': features[val_indices], 'labels': labels[val_indices]}
    return train, val


def triple_cross_validate(features: list, labels: list, num_labels: int):
    """
    Perform the triple cross validation

    :param: features: Data
    :param: labels: List of the labels corresponding to the rows of the features
    :param: num_labels: Number of unique labels
    :return: outer_best: Accuracy of best performing model
    :return: outer_accuracies: Accuracies of the models in the outer loop
    :return: middle_accuracies: Accuracies of the models in the middle loop
    :return: inner_accuracies: Accuracies of the models in the inner loop
    """
    ensemble = BestEnsemble(len(features[1,:]), num_labels)

    start_time = time.time()
    outer_fold, middle_fold, inner_fold = OUTER_FOLD, MIDDLE_FOLD, INNER_FOLD
    outer_accuracies, middle_accuracies, inner_accuracies = [], [], []
    outer_best = {'accuracy': 0, 'model': None, 'selector': None}

    index_stratified,count_rem1 = stratification(labels,outer_fold,num_labels)

    features = features[index_stratified]
    labels = labels[index_stratified]

    # Outer fold, used for accuracy validation of best selector/classifier pairs
    for outer_i in range(0, outer_fold):

        outer_train, outer_val = slice_data(features, labels, outer_fold, outer_i)
        middle_best = {'accuracy': 0, 'model': None, 'selector': None}

        index_stratified,count_rem2 = stratification(outer_train['labels'],middle_fold,num_labels)
        outer_train['features'] = outer_train['features'][index_stratified]
        outer_train['labels'] = outer_train['labels'][index_stratified]

        # Middle fold, used for selecting the optimal selector
        for middle_i in range(0, middle_fold):
            for selector_i in range(0, len(selectors)):

                middle_train, middle_val = slice_data(outer_train['features'], outer_train['labels'], middle_fold,
                                                      middle_i)
                selector = list(selectors.values())[selector_i]()
                selected_indices = selector.select_features(middle_train['features'], middle_train['labels'])
                inner_best = {'accuracy': 0, 'model': None}

                index_stratified,count_rem3 = stratification(middle_train['labels'],inner_fold,num_labels)
                middle_train['features'] = middle_train['features'][index_stratified]
                middle_train['labels'] = middle_train['labels'][index_stratified]

                # Inner fold, used for selecting the optimal classifier
                for inner_i in range(0, inner_fold):
                    for classifier_i in range(0, len(classifiers)):

                        inner_train, inner_val = slice_data(middle_train['features'], middle_train['labels'],
                                                            inner_fold,
                                                            inner_i)
                        classifier = list(classifiers.values())[classifier_i](len(selected_indices), num_labels)

                        print('[inner] Training %s / %s' % (classifier.__class__.__name__, selector.__class__.__name__))
                        train_acc = classifier.train(inner_train['features'][:, selected_indices],
                                                     inner_train['labels'])
                        accuracy = classifier.predict(inner_val['features'][:, selected_indices], inner_val['labels'])
                        inner_accuracies.append({'train_accuracy': train_acc, 'accuracy': accuracy,
                                  'model': list(classifiers.values())[classifier_i],
                                  'model_name':list(classifiers.keys())[classifier_i],
                                  'selector': selector.__class__, 'indices': selected_indices})

                        del classifier

                inner_best = get_best_performing(inner_accuracies)

                ensemble.add_combination(inner_best)

                # Calculate and save accuracy of best classifier for current feature selector
                classifier = inner_best['model'](len(selected_indices), num_labels)
                print('[middle] Training %s / %s' % (classifier.__class__.__name__, selector.__class__.__name__))
                train_acc = classifier.train(middle_train['features'][:, selected_indices], middle_train['labels'])
                accuracy = classifier.predict(middle_val['features'][:, selected_indices], middle_val['labels'])

                middle_accuracies.append({'train_accuracy': train_acc, 'accuracy': accuracy,
                                          'model': inner_best['model'],'model_name':inner_best['model_name'],
                                          'selector': inner_best['selector'], 'indices': selected_indices})

                cur_time = time.time()
                pct_complete = (
                    ((outer_i+1) * MIDDLE_FOLD * len(selectors) + (middle_i+1) * len(selectors) + (selector_i+1)) /
                    (OUTER_FOLD * MIDDLE_FOLD * len(selectors) + MIDDLE_FOLD * len(selectors) + len(selectors)))
                print('======== Progress: %f%% (estimated time remaining %s) ========' % (
                    pct_complete * 100, datetime.timedelta(seconds=(cur_time-start_time)/pct_complete)
                ))
                del classifier, selector

        # ensemble middle loop
        for middle_i in range(0,middle_fold):

            middle_train,middle_val = slice_data(outer_train['features'],outer_train['labels'],middle_fold,
                                                     middle_i)
            print('[middle] Training Best ensemble')
            train_acc = ensemble.train(middle_train['features'],middle_train['labels'])
            accuracy = ensemble.predict(middle_val['features'],middle_val['labels'])

            middle_accuracies.append({'train_accuracy': train_acc,'accuracy': accuracy,
                                      'model': list(ensembles.values())[0],'model_name':list(ensembles.keys())[0],
                                      'selector': AllSelector, 'indices': np.array(range(len(middle_train['features'][0])))})

        middle_best = get_best_performing(middle_accuracies)

        # Calculate and save accuracy of best feature selector / classifier pair
        selector = middle_best['selector']()
        selected_indices = selector.select_features(outer_train['features'], outer_train['labels'])

        classifier = middle_best['model'](len(selected_indices), num_labels)

        if middle_best['model_name'] == 'best_ens':
            classifier.add_combinations_list(middle_accuracies)
            selected_indices = middle_best['indices']

        print('[outer] Training %s / %s' % (classifier.__class__.__name__, selector.__class__.__name__))
        train_acc = classifier.train(outer_train['features'][:, selected_indices], outer_train['labels'])
        accuracy = classifier.predict(outer_val['features'][:, selected_indices], outer_val['labels'])
        outer_accuracies.append({'train_accuracy': train_acc, 'accuracy': accuracy, 'model': middle_best['model'],
                                 'model_name':middle_best['model_name'],
                  'selector': middle_best['selector'], 'indices': selected_indices})

        del classifier, selector

    outer_best = get_best_performing(outer_accuracies)
    print(outer_best)
    print('{} samples has been removed during folds stratification'.format(count_rem1 + count_rem2 + count_rem3))
    return outer_best, outer_accuracies, middle_accuracies, inner_accuracies



def make_faded(colorcode):
    """
    Takes a hex RGB color code, and returns a faded (less saturation) version of it.
    :param colorcode: Hex RGB string (e.g. #AABBCC)
    :return: Hex RGB string (e.g. #AABBCC)
    """
    r, g, b = ImageColor.getrgb(colorcode)
    # r = np.random.randint(10,245)
    # g = np.random.randint(30,225)
    # b = np.random.randint(20,215)
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    l = min([l * 1.5, 1.0])
    s *= 0.4
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return '#%02X%02X%02X' % (int(r * 255), int(g * 255), int(b * 255))


def plot_accuracies(accuracies: list, title='Accuracies', hist_title='Selected features'):
    """
    Plots the training and testing accuracy of different model and feature selector combinations
    :param accuracies: List of accuracies
    """
    plt.figure()

    selected_indices = []

    field_names = ['accuracy', 'train_accuracy']
    for index, field_name in enumerate(field_names):
        grouped = {}
        for entry in accuracies:
            model, selector, accuracy, indices = entry['model'], entry['selector'], entry[field_name], entry['indices']
            model_name = entry['model_name']
            if model_name == 'best_ens':
                selected_indices = []

            key = '%s / %s' % (selector.__name__.replace('Selector', ''), model.__name__.replace('Classifier', ''))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(accuracy)
            selected_indices += list(indices)

        means = [np.mean(group) for group in grouped.values()]
        stds = [np.std(group) for group in grouped.values()]
        names = ['%s / %d' % (key, len(grouped[key])) for key in grouped.keys()]
        available_colours = ['#0082c8', '#3cb44b', '#ffe119', '#f58231', '#e6194b', '#911eb4', '#d2f53c',
                             '#fabebe', '#008080', '#aa6e28', '#800000', '#ffd8b1']
        selectors = [name.split(' / ')[0] for name in names]
        unique_selectors = list(set(selectors))
        colours = [available_colours[unique_selectors.index(selector) % len(available_colours)] for selector in
                   selectors]
        if index > 0:
            colours = [make_faded(colour) for colour in colours]
        ticks = [num * len(field_names) + index for num in range(0, len(means))]

        plt.bar(ticks, means, yerr=stds, color=colours)
        plt.xticks(ticks, names, rotation=90, fontsize=6)

    plt.xlabel('Selector/classifier pairs')
    plt.ylabel('Validation accuracy')
    plt.title(title)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig('results/%s.png'%title)

    unique, counts = np.unique(selected_indices, return_counts=True)
    x = range(0, len(unique))
    plt.bar(x, counts)
    plt.xticks(x, unique, rotation=90, fontsize=6)
    plt.xlabel('Selected feature index')
    plt.ylabel('Count')
    plt.title(hist_title)
    plt.savefig('results/%s.png'%hist_title)


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
        #if not (N_SAMPLES % OUTER_FOLD == 0):
        #    print('OUTER_FOLD constant is not appropriate.')
        #    print('Script execution is aborted after %.8s s.' % (time.time() - START_TIME))
        #    sys.exit()

        # Select model to run, based on command line parameter
        feature_length, num_unique_labels = features.shape[1], len(set(labels))

        # give standard output
        print('Triple cross-validation with %i-fold and subsequent %i-fold and %i-fold splits is initiated.' %
              (OUTER_FOLD, len(selectors.keys()), len(classifiers.keys())))

        best, outer_acc, middle_acc, inner_acc = triple_cross_validate(features, labels, num_unique_labels)
        np.save('cache/best.npy', best)
        np.save('cache/outer_acc.npy', outer_acc)
        np.save('cache/middle_acc.npy', middle_acc)
        np.save('cache/inner_acc.npy', inner_acc)

        # TODO: Save model as *.pkl USING sklearn.joblib() ?
        # Train one last time on entire dataset
        model = create_final_model(best, best['selector'], features, labels, num_unique_labels, middle_acc)
    else:
        best, outer_acc, middle_acc, inner_acc = np.load('cache/best.npy').item(), np.load('cache/outer_acc.npy'), \
                                     np.load('cache/middle_acc.npy'), np.load('cache/inner_acc.npy')

    plot_accuracies(inner_acc, 'Inner fold accuracies')
    plot_accuracies(middle_acc, 'Middle fold accuracies')
    plot_accuracies(outer_acc, 'Outer fold accuracies')

    print('Triple-CV was finished.')
    print('Best performing pair (%f%%): %s / %s' % (best['accuracy'], best['selector'], best['model']))


# EXECUTION
if __name__ == '__main__':
    main()
    print('\nFinished: The script was successfully executed in %.8s s.' % (time.time() - START_TIME))
