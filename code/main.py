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

# CONSTANTS
START_TIME = time.time()
BEANS_CONSTANT = 69
N_SAMPLES = 100  # number of samples (patients)
N_VARIABLES = 2834  # number of chromosomal locations
OUTER_FOLD = 4  # OUTER_FOLD-fold CV (outer loop) for triple-CV (Wessels, 2005: 3-fold)
INNER_FOLD = 5  # INNER_FOLD-fold CV (inner loop) for triple-CV (Wessels, 2005: 10-fold)


# FUNCTION DEFINITIONS
def main():
	print('Script execution was initiated.')

	# SETTING THE SEED FOR REPRODUCIBILITY
	np.random.seed(0)

	# IMPORT OF RAW DATA AS NP.ARRAY USING PANDAS
	# import of train_clinical.txt, samples x 2 (sample name, label)
	# Sample example (100 in total, sample name): 'Array.129'
	# Subgroups (3 in total, label): 'HER2+', 'HR+' and 'Triple Neg'
	train_clinical = pd.read_csv('../data/Train_clinical.txt', sep='\t').values
	print('Data set "train_clinical" was loaded (%i rows and %i cols).' % (
		train_clinical.shape[0], train_clinical.shape[1]))
	# import and transpose of train_call.txt, samples x variables
	train_call = np.transpose(
		pd.read_csv('../data/Train_call.txt', sep='\t', usecols=range(4, 4 + N_SAMPLES)).values.astype('float32'))
	print('Data set "train_call" was loaded (%i rows and %i cols).' % (train_call.shape[0], train_call.shape[1]))
	""" The order in both np.arrays is the same as in the original files, which means that the label (output) \\
	train_clinical[a, 1] is the wanted prediction for the data (features) in train_call[a, :]"""

	# DATA PRE-PROCESSING
	# yet, empty

	# TRIPLE CV (with random sampling without replacement) (similar to Wessels, 2005)
	# (ALTERNATIVE: WITH REPLACEMENT, then other OUTER_FOLD AND INNER_FOLD are allowed)
	# test if provided constants INNER_FOLD and OUTER_FOLD are allowed
	if not (N_SAMPLES % OUTER_FOLD == 0 and N_SAMPLES / OUTER_FOLD % INNER_FOLD == 0):
		print('INNER_FOLD and OUTER_FOLD constants are not appropriate.')
		print('Script execution is aborted after %.8s s.' % (time.time() - START_TIME))
		sys.exit()
	# give standard output
	print ('Triple-CV with %i-fold and subsequent %i-fold split is initiated.' % (OUTER_FOLD, INNER_FOLD))
	# run actual triple-CV using accuracy as the evaluation metric
	train_accuracy_outer_loop_list, val_accuracy_outer_loop_list = [], []
	outer_chunk_size = int(N_SAMPLES / OUTER_FOLD) # number of samples provided after first split
	inner_chunk_size = int(N_SAMPLES / OUTER_FOLD / INNER_FOLD) # number of samples provided after second split
	# creation of an np.array of OUTER_FOLD x outer_chunk_size, giving the indices of the samples
	outer_chunks = np.random.choice(N_SAMPLES, size=(OUTER_FOLD, outer_chunk_size), replace=False)
	for outer_iterator in range(0, OUTER_FOLD):
		train_accuracy_inner_loop_list, val_accuracy_inner_loop_list = [], []
		# provide the indices for the data which are projected to the inner loop
		cur_outer_chunk = outer_chunks[outer_iterator, :]
		inner_chunks = np.random.choice(cur_outer_chunk, size=(INNER_FOLD, inner_chunk_size), replace=False)
		for inner_iterator in range(0, INNER_FOLD):
			# SELECT TRAIN (training set)
			train_selector = [not index == inner_iterator for index in range(0, INNER_FOLD)]
			train_indices = inner_chunks[train_selector, :]
			train_features = np.reshape(train_call[train_indices, :],
										(int(N_SAMPLES/OUTER_FOLD-N_SAMPLES/OUTER_FOLD/INNER_FOLD), N_VARIABLES))
			train_labels = np.reshape(train_clinical[train_indices, 1],
					   (int(N_SAMPLES / OUTER_FOLD - N_SAMPLES / OUTER_FOLD / INNER_FOLD), 1))

			# SELECT VAL (validation set)
			val_indices = inner_chunks[inner_iterator, :]
			val_features = train_call[val_indices, :]  # inner_chunk_size x N_VARIABLES
			val_labels = train_clinical[val_indices, 1]  # inner_chunk_size x 1

			# TRAIN THE MODEL: CALL TRAIN FUNCTION FROM MODULE OF CLASSIFIER WITH RETURN OF TRAINING LOSS (acc.)
			"""
			train_features is a matrix of shape (number_of_train_samples, dimensions) =
			(N_SAMPLES/OUTER_FOLD-N_SAMPLES/OUTER_FOLD/INNER_FOLD, N_VARIABLES)
			train_labels is a matrix of shape (number_of_train_samples, 1) =
			(N_SAMPLES/OUTER_FOLD-N_SAMPLES/OUTER_FOLD/INNER_FOLD, N_VARIABLES)
			"""
			# train_accuracy_inner_loop = train_(train_features, train_labels) # remove #
			# train_accuracy_inner_loop_list.append(train_accuracy_inner_loop) # remove #

			# PREDICT: CALL PREDICT FUNCTION FROM TRAINED CLASSIFIER WITH RETURN OF PREDICTION LOSS (acc.)
			"""
			val_features is a matrix of shape (number_of_val_samples, dimensions) =
			(N_SAMPLES/OUTER_FOLD/INNER_FOLD, N_VARIABLES)
			val_labels is a matrix of shape (number_of_val_samples, 1) =
			(N_SAMPLES/OUTER_FOLD/INNER_FOLD, N_VARIABLES)
			"""
			# val_accuracy_inner_loop = predict(val_features, val_labels) # remove #
			# val_accuracy_inner_loop_list.append(val_accuracy_inner_loop) # remove #
		train_accuracy_outer_loop_list.append(train_accuracy_inner_loop_list)
		val_accuracy_outer_loop_list.append(val_accuracy_inner_loop_list)

	# training_loss is np.array of shape (OUTER_FOLD, INNER_FOLD)
	# training_loss = np.array(train_accuracy_outer_loop_list) # training accuracies # remove #
	# training_loss_mean = np.mean(np.reshape(training_loss, (OUTER_FOLD*INNER_FOLD, 1))) # remove #

	# validation loss is np.array of shape (OUTER_FOLD, INNER_FOLD)
	# validation_loss = np.array(val_accuracy_outer_loop_list) # validation accuracies # remove #
	# validation_loss_mean = np.mean(np.reshape(validation_loss, (OUTER_FOLD*INNER_FOLD, 1))) # remove #

	print('Triple-CV was finished.')
	# print('Training loss (accuracy): %.4f.' % training_loss_mean) # remove #
	# print(colored('Validation loss (accuracy): %.4f.', 'green') % validation_loss_mean + " You 'mirin, bra?") # remove #
	print('The distribution of the evaluation metric (accuracy) is being plotted.')

	# DISPLAY OF TRAINING AND VALIDATION LOSS (acc.)
	""" # remove comment
	train_accs = np.reshape(train_loss, (OUTER_FOLD*INNER_FOLD, 1))
	val_accs = np.reshape(validation_loss, (OUTER_FOLD*INNER_FOLD, 1))
	plt.figure()
    plt.plot(train_accs, [0 for a in range(0, len(train_accs))], alpha = 0.4, label='training accuracies')
    plt.plot(val_accs, [1 for a in range(0, len(val_accs))], alpha = 0.4, label='validation accuracies')
    plt.xlabel('Order parameter')
    plt.ylabel('Accuracy')
    plt.title('Distribution of accuracies in three-fold CV')
    plt.legend()
    plt.show()
	"""

# SAVE MODEL AS *.pkl USING sklearn.joblib()


# EXECUTION
if __name__ == '__main__':
	main()
	print('\nFinished: The script was successfully executed in %.8s s.' % (time.time() - START_TIME))


# END OF FILE