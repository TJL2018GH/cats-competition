# cats-competition
3-state classifier from translational bioinformatics: Copy number data is used to classify tumor types.
Data is high-dimensional (100 samples, >2.8k variables), therefore sparse classifiers have to be utilized.

*To-Do*
Implement sklearn.pipeline.Pipeline to combine feature selection and training of the classifier, as exemplified in the snippet below, where LinearSVS with L1-norm regularization and removal of low scoring features is used to select features, and subsequently a random forest classifiers is trained on the reduced features space.
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)

*Think space* (general idea communicated in-vivo group discussion)
The following ideas can be implemented with the sklearn pipeline approach.

The models applied to the raw data set can be subdivided into 2 parts: part A (feature selection, feature extraction, dimension reduction) & part B (classifier model with appropriate learner). We could apply a learner to explore the model space of all (part A, part B) combinations either entirely (grid search) or heuristically (evolutionary computing), depending on the number of combinations (size of the search space).

part A: Top-k cross-3-state statistically signficant features, unsupervised dimension reduction to k features by PCA, supervised dimension reduction to k features by (linear) discriminant analysis;
Different methods can be found in publications (e. g. https://arxiv.org/pdf/1403.2877.pdf).

part B: multivariate linear classifier (sklearn), k-NN (sklearn), decision tree classifier (sklearn), nearest mean classifier, random forest, probabilitistic classifiers, feedforward neural network (NN), more advanced NNs (if sample size is deemed sufficiently large for training);
The methods shall be tested with different distance metrics (e. g. cosine distance for the nearest mean classifier).

Constant k is a hyper-parameter.

The observed values in our dataset for each variable are discrete, being in the interval {-2, -1, 0, 1, 2} (please verify). Therefore, the data is suitable for comparison using 'hamming' (particularly, with a mean hamming-centroid per class), 'canberra' and 'braycurtis' distance.

Regularization (keep the classifier 'simple', and thereby prevents overfitting)
Included a regularization term in the loss function to prevent overfitting in our highly undetermined problem (n_vars >> n_samples). Examples: L0-norm (addition of number of non-zero weights to loss function), L1-norm (LASSO with LS-loss), L2-norm (Tikhonov-regularizer, ridge regression), elastic net (combination of L1 and L2).

*Part A: Feature Selection/Extraction*
Library: sklearn.feature_selection
Selection:
  - Removal of features with low variance (VarianceThreshold)
  - Keeping the k-best features based on univariate statistics (SelectKBest)
      scoring functions: f_classif (F-score), mutual_info_classif (nonpar., mutual information) (both have adv. and disadv.)
  - Recursive feature selection (RFE): iterated removal of the feature of lowest importance while constructing the classifier (can be only used if estimator provides a measure of feature importance in coef_ or feature_importances attribute)
  - Non-recursive feature selection (SelectFromModel): one hit removal of badly scoring features (can be only used if estimator provides a measure of feature importance in coef_ or feature_importances attribute)
  - L1-norm based: in combination with the classifiers linear_model.LogisticRegression and svm.LinearSVC
  - Tree-based: Use SelectFromModel (see above) in combination with decision tree (sklearn.tree) or decision forest (sklearn.ensemble);
And custom or other approaches (e. g. from literature).
   
*Our features*
Log2-fold copy number (CN) changes (hereafter: 'values') for 2,834 (exclusive, non-exhaustive, differently-sized) chromosomal segments (hereafter: 'segments') covering the 22 autosomes and the X-chromosome (caveat: all samples are derived from female patients), for each of which we know the 'chromosome', 'start position', and 'end position'.
Exclusive, but non-exhaustive, and equally (as experimentally possible) spaced chromosomal stretches.
Log2-fold changes (values): '-1': Loss, '0': No Change, '1': Duplication, '2': Tetraplication;

*Approaches in Literature*
Problems of the methods: filtering out of useful information by feature selection, neglecting inherent structure (correlation, see: start and end position of duplication or deletion) of aCGH data

Rapaport et al., 2008:
Chin et al., 2006: feature selection (small number of genes) & kNN; two-state: estrogen+ vs. estrogen-; (acc.=76%, in balanced set)
Jones et al., 2004: feature selection (reduction to gain/loss at chromosome arm resolution) & nearest centroid classification; muli-state: breast tumor grade; (acc. larger than for Chin et al., 2006)


