# cats-competition
3-state classifier from translational bioinformatics: Copy number data is used to classify tumor types.

Think space:
The models applied to the raw data set can be subdivided into 2 parts: part A (feature selection, feature extraction, dimension reduction) & part B (classifier model with appropriate learner). We could apply a learner to explore the model space of all (part A, part B) combinations either entirely (grid search) or heuristically (evolutionary computing), depending on the number of combinations (size of the search space).

part A: Top-k cross-3-state statistically signficant features, unsupervised dimension reduction to k features by PCA, supervised dimension reduction to k features by (linear) discriminant analysis;
Different methods can be found in publications (e. g. https://arxiv.org/pdf/1403.2877.pdf).

part B: multivariate linear classifier (sklearn), k-NN (sklearn), decision tree classifier (sklearn), nearest mean classifier, random forest, probabilitistic classifiers, feedforward neural network (NN), more advanced NNs (if sample size is deemed sufficiently large for training);
The methods shall be tested with different distance metrics (e. g. cosine distance for the nearest mean classifier).

Constant k is a hyper-parameter.

The observed values in our dataset for each variable are discrete, being in the interval {-2, -1, 0, 1, 2} (please verify). Therefore, the data is suitable for comparison using 'hamming' (particularly, with a mean hamming-centroid per class), 'canberra' and 'braycurtis' distance.

Regularization (keep the classifier 'simple', and thereby prevents overfitting)
Included a regularization term in the loss function to prevent overfitting in our highly undetermined problem (n_vars >> n_samples). Examples: L0-norm (addition of number of non-zero weights to loss function), L1-norm (LASSO with LS-loss), L2-norm (Tikhonov-regularizer, ridge regression), elastic net (combination of L1 and L2).
