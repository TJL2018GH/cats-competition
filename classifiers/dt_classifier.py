# IMPORTS

# NOT YET IMPLEMENTED
# Please announce if you implement it.
# Ultima ratio: Ben provides himself for implementation

from classifiers.base_classifier import BaseClassifier
from sklearn import tree


class KNearestNeighborsClassifier(BaseClassifier):
    # encoder = LabelBinarizer()

    def __init__(self, feature_length, num_classes):
        super().__init__(feature_length, num_classes)
        self.num_classes = num_classes

        ###
        # BUILD YOUR MODEL
	self.model = tree.DecisionTreeClassifier(criterion='gini', random_state=0)
	
        ###

    def train(self, features, labels):
        """
        Using a set of features and labels, trains the classifier and returns the training accuracy.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to train to predict
        :return: Prediction accuracy, as a float between 0 and 1
        """
        labels = self.labels_to_categorical(labels)
	self.model.fit(features, labels)
	train_accuracy = score(self.model.predict(features), labels)
	
	# save depiction of the tree model
	"""
	dot -Tpng dt_classifier.dot -o dt_classifier.png
	"""
	tree.export_graphviz(self.model, out_file='dt_classifier.dot')

	# get parameters by get_params()
        return train_accuracy

    def predict(self, features, labels):
        """
        Using a set of features and labels, predicts the labels from the features,
        and returns the accuracy of predicted vs actual labels.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to test prediction accuracy on
        :return: Prediction accuracy, as a float between 0 and 1
        """
        labels = self.labels_to_categorical(labels)
	self.model.predict(features)
	test_accuracy = score(self.model.predict(features), labels)
        return test_accuracy

    def labels_to_categorical(self, labels):
        _, IDs = unique(labels, return_inverse=True)
        return to_categorical(IDs, num_classes=self.num_classes)
